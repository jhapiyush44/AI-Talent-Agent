"""
agent.py — Orchestration layer.

Ties together:
  - JD parsing
  - Resume loading (with retry / quarantine)
  - Pinecone RAG upsert + semantic retrieval
  - Hybrid scoring
  - Interest simulation (LLM, Top-K only)
  - Final ranking
"""

import os
import json
import logging
import hashlib
from pathlib import Path

from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

from .jd_parser    import parse_jd
from .resume_parser import parse_resumes
from .scorer       import compute_match_score, decide
from .vector_store import get_vector_store

logger = logging.getLogger(__name__)

# ── Model config — change here or set GEMINI_MODEL env var ───────────────────
# Free-tier stable models as of May 2026: gemini-2.5-flash, gemini-2.5-flash-lite
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

CACHE_FILE     = "resumes_cache.json"
INTEREST_TOP_K = 5   # Only call LLM for interest simulation on top-K

INTEREST_PROMPT = """You are a senior recruiter's AI assistant.

Given the job description excerpt and candidate profile below, write a brief,
realistic 2-3 sentence simulated interest response FROM the candidate's
perspective — as if they just read the JD.

Job Description (excerpt):
{jd_excerpt}

Candidate:
Name: {name}
Skills: {skills}
Experience: {exp} years
Summary: {summary}

Return ONLY the simulated response text. No headers, no JSON.
"""

# ── Cache helpers ─────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache(cache: dict):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.warning("Cache save failed: %s", e)

# ── Embedding helper ──────────────────────────────────────────────────────────

_embed_model: SentenceTransformer = None

def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        logger.info("🔄  Loading embedding model...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def embed(text: str) -> list:
    return _get_embed_model().encode(text, normalize_embeddings=True).tolist()

# ── Interest simulation ───────────────────────────────────────────────────────

def _simulate_interest(candidate: dict, jd_text: str, model) -> tuple[float, str, str]:
    """Returns (interest_score, simulated_response, reason)."""
    try:
        prompt = INTEREST_PROMPT.format(
            jd_excerpt=jd_text[:800],
            name=candidate.get("name", ""),
            skills=", ".join(candidate.get("skills", [])[:15]),
            exp=candidate.get("experience_years") or "N/A",
            summary=candidate.get("summary", "")[:300],
        )
        response = model.generate_content(prompt)
        text = response.text.strip()

        # Heuristic interest score from response sentiment
        positive_signals = ["excited", "great opportunity", "perfect fit", "love", "ideal",
                            "thrilled", "interested", "passion", "well-aligned", "strong match"]
        neutral_signals  = ["consider", "open to", "would like", "potentially"]
        hits_pos = sum(1 for s in positive_signals if s in text.lower())
        hits_neu = sum(1 for s in neutral_signals  if s in text.lower())

        if hits_pos >= 2:
            score = 0.85
        elif hits_pos == 1:
            score = 0.70
        elif hits_neu >= 1:
            score = 0.55
        else:
            score = 0.40

        return score, text, "LLM simulated"

    except Exception as e:
        logger.warning("Interest simulation failed: %s", e)
        return 0.5, "Unable to simulate interest at this time.", "Fallback"

# ── Main run function ─────────────────────────────────────────────────────────

def run_agent(
    jd_text: str,
    resume_dir: str,
    top_k: int = 10,
    job_id: str = None,
) -> dict:
    """
    Full agent pipeline. Returns result dict ready for API response.
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY (or GEMINI_API_KEY) not set in environment")

    client = genai.Client(api_key=api_key)

    # Thin wrapper so parsers receive a .generate_content(prompt) compatible object
    class _ModelShim:
        def generate_content(self, prompt: str):
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return response

    model = _ModelShim()

    # ── 1. Parse JD ───────────────────────────────────────────────────────────
    logger.info("📋  Parsing job description...")
    jd = parse_jd(jd_text, model)

    # ── 2. Load + parse resumes ───────────────────────────────────────────────
    logger.info("📂  Loading resumes from: %s", resume_dir)
    cache = _load_cache()
    candidates = parse_resumes(resume_dir, model, cache)
    _save_cache(cache)
    logger.info("👥  Loaded %d candidate(s)", len(candidates))

    if not candidates:
        return {"error": "No resumes found or all failed to parse.", "top_candidates": []}

    # ── 3. Embed JD ───────────────────────────────────────────────────────────
    jd_summary = (
        " ".join(jd.get("required_skills", []))
        + " "
        + " ".join(jd.get("keywords", []))
        + " "
        + jd_text[:500]
    )
    jd_embedding = embed(jd_summary)

    # ── 4. Pinecone RAG ───────────────────────────────────────────────────────
    vs = get_vector_store()
    ns = job_id or hashlib.md5(jd_text[:200].encode()).hexdigest()[:12]

    rag_scores: dict[str, float] = {}
    if vs.enabled:
        vs.upsert_candidates(candidates, embed_fn=embed, namespace=ns)
        rag_matches = vs.query_similar(jd_summary, embed_fn=embed, top_k=top_k * 2, namespace=ns)
        for m in rag_matches:
            # Key by source_file if available, fall back to name
            key = m.get("source_file") or m.get("name", "").lower()
            rag_scores[key] = m.get("rag_score", 0.0)
        logger.info("🔍  RAG scores retrieved for %d candidates", len(rag_scores))

    # ── 5. Score all candidates ───────────────────────────────────────────────
    scored = []
    for c in candidates:
        logger.info("📊  Scoring: %s", c.get("name", "?"))
        rag_key = c.get("_source_file") or c.get("name", "").lower()
        rag = rag_scores.get(rag_key)
        match_score, explanation = compute_match_score(c, jd, jd_embedding, embed, rag)
        scored.append({
            **c,
            "match_score": match_score,
            "explanation": explanation,
            "interest_score": 0.5,
            "simulated_response": "",
            "interest_reason": "Pending",
        })

    # ── 6. Rank ───────────────────────────────────────────────────────────────
    scored.sort(key=lambda x: x["match_score"], reverse=True)
    top_candidates = scored[:top_k] if top_k < 100 else scored

    # ── 7. Interest simulation (Top-K only) ───────────────────────────────────
    for c in top_candidates[:INTEREST_TOP_K]:
        logger.info("💬  Simulating interest: %s", c.get("name", "?"))
        i_score, i_resp, i_reason = _simulate_interest(c, jd_text, model)
        c["interest_score"]    = i_score
        c["simulated_response"] = i_resp
        c["interest_reason"]   = i_reason

    # ── 8. Final score + decision ─────────────────────────────────────────────
    # Interest simulation is informational only — NOT part of the ranking score.
    # It's synthetic LLM output that always sounds positive; including it inflates scores.
    for c in top_candidates:
        c["final_score"] = c["match_score"]   # match_score already includes tiebreakers
        c["decision"]    = decide(c["final_score"])

    # ── 9. Clean up internal fields before returning ──────────────────────────
    clean = []
    for c in top_candidates:
        clean.append({
            "name":               c.get("name", "Unknown"),
            "email":              c.get("email"),
            "experience_years":   c.get("experience_years"),
            "skills":             c.get("skills", []),
            "match_score":        c["match_score"],
            "interest_score":     c["interest_score"],
            "final_score":        c["final_score"],
            "decision":           c["decision"],
            "explanation":        c["explanation"],
            "simulated_response": c["simulated_response"],
            "interest_reason":    c["interest_reason"],
            "rag_boosted":        (c.get("_source_file") or c.get("name", "").lower()) in rag_scores,
        })

    return {
        "top_candidates": clean,
        "total_evaluated": len(scored),
        "jd_required_skills": jd.get("required_skills", []),
        "jd_optional_skills": jd.get("optional_skills", []),
        "pinecone_enabled": vs.enabled,
    }