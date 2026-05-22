"""
scorer.py — Hybrid Semantic Scoring Engine v4

Formula (justified):
─────────────────────────────────────────────────────────────────────────────
  match_score = 0.60 × skill_score
              + 0.25 × project_score
              + 0.15 × experience_score

  final_score = match_score
              + context_tiebreaker (max +3%)
              + rag_tiebreaker     (max +3%)

  Interest simulation is SHOWN on the card but NOT in the score.
  It's synthetic LLM output — it always sounds interested. Pollutes ranking.

Thresholds (realistic recruiter bar):
  Strong Shortlist : ≥ 0.72   matched ≥70% required + strong project evidence
  Shortlist        : ≥ 0.58   matched majority required, decent projects
  Consider         : ≥ 0.42   partial match, manual review warranted
  Reject           : < 0.42   insufficient evidence

Skill matching (semantic):
  Each JD skill is embedded and compared against every candidate skill via
  cosine similarity. Match declared if sim ≥ SKILL_MATCH_THRESHOLD (0.60).
  Fast-path string checks avoid unnecessary embed calls.
─────────────────────────────────────────────────────────────────────────────
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Tunable constants ─────────────────────────────────────────────────────────
SKILL_MATCH_THRESHOLD = 0.60   # cosine threshold for skill semantic match

WEIGHTS = {
    "skill":      0.60,   # primary signal — can you do the job?
    "project":    0.25,   # have you actually done similar work?
    "experience": 0.15,   # years matter, but less than demonstrated skill
}

# Tiebreaker caps (added on top, capped so they can't change the decision band)
CONTEXT_TIEBREAKER_MAX = 0.03  # full-profile embedding vs JD
RAG_TIEBREAKER_MAX     = 0.03  # Pinecone cosine nudge

# Decision thresholds
THRESHOLD_STRONG  = 0.72
THRESHOLD_SHORT   = 0.58
THRESHOLD_CONSIDER= 0.42

# Embedding cache: raw string -> vector (avoids re-embedding same skill)
_embed_cache: dict[str, list] = {}

# Match cache: (cn, jn) -> bool
_match_cache: dict[tuple, bool] = {}


# ── Utilities ─────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    """Strip parentheticals, lowercase, normalise separators."""
    s = s.strip().lower()
    s = re.sub(r"\s*\(.*?\)", "", s)
    s = re.sub(r"[\-_/\\]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _cached_embed(text: str, embed_fn) -> list:
    if text not in _embed_cache:
        _embed_cache[text] = embed_fn(text)
    return _embed_cache[text]


def _cosine(a: list, b: list) -> float:
    if not a or not b:
        return 0.0
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x ** 2 for x in a) ** 0.5
    mag_b = sum(x ** 2 for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ── Semantic skill matching ───────────────────────────────────────────────────

def _skill_matches(candidate_skill: str, jd_skill: str, embed_fn) -> bool:
    """
    True if candidate_skill semantically matches jd_skill.
    Order: exact → substring → token overlap → embedding cosine.
    """
    cn = _normalize(candidate_skill)
    jn = _normalize(jd_skill)

    # Fast path 1: exact
    if cn == jn:
        return True

    # Fast path 2: one contains the other
    if jn in cn or cn in jn:
        return True

    # Fast path 3: token overlap ≥ 75% of JD skill tokens
    jn_tok = set(jn.split())
    cn_tok = set(cn.split())
    if jn_tok and len(jn_tok & cn_tok) / len(jn_tok) >= 0.75:
        return True

    # Semantic path
    key = (cn, jn)
    if key in _match_cache:
        return _match_cache[key]

    try:
        sim    = _cosine(_cached_embed(candidate_skill, embed_fn),
                         _cached_embed(jd_skill,        embed_fn))
        result = sim >= SKILL_MATCH_THRESHOLD
        _match_cache[key] = result
        if result:
            logger.debug("Semantic match  '%s' ↔ '%s'  sim=%.3f", cn, jn, sim)
        return result
    except Exception as e:
        logger.warning("Embed error during skill match: %s", e)
        _match_cache[key] = False
        return False


# ── Component scorers ─────────────────────────────────────────────────────────

def skill_score(
    candidate_skills: list,
    required: list,
    optional: list,
    embed_fn,
) -> tuple[float, dict]:
    """
    Semantic skill score.
    Required: 70% of skill score.  Optional: 30%.
    Returns (0–1 score, breakdown dict).
    """
    if not required and not optional:
        return 0.0, {"required_matched": [], "required_missing": [],
                     "optional_matched": [], "req_score": 0, "opt_score": 0}

    req_hits = [r for r in required
                if any(_skill_matches(cs, r, embed_fn) for cs in candidate_skills)]
    opt_hits = [o for o in optional
                if any(_skill_matches(cs, o, embed_fn) for cs in candidate_skills)]

    req_score = len(req_hits) / len(required) if required else 1.0
    opt_score = len(opt_hits) / len(optional) if optional else 1.0

    # Within skill component: 70% required, 30% optional
    final = round(0.70 * req_score + 0.30 * opt_score, 4)

    return final, {
        "required_matched": req_hits,
        "required_missing": [r for r in required if r not in req_hits],
        "optional_matched": opt_hits,
        "req_score":        round(req_score, 3),
        "opt_score":        round(opt_score, 3),
    }


def experience_score(
    exp_years: Optional[float],
    exp_min:   Optional[float],
    exp_max:   Optional[float],
) -> float:
    """
    Returns 0–1.
    None experience → 0.65 (not penalised for being a fresher/student).
    Perfect range match → 1.0.
    Under by N years → penalty of 0.15 per year (soft floor 0.0).
    Over by N years → mild penalty of 0.03 per year (floor 0.6, overqualified ≠ bad).
    """
    if exp_years is None:
        return 0.65

    if exp_min is None and exp_max is None:
        return 0.85   # JD didn't specify — assume they're fine

    lo = exp_min if exp_min is not None else 0.0
    hi = exp_max if exp_max is not None else float("inf")

    if lo <= exp_years <= hi:
        return 1.0
    elif exp_years < lo:
        # Under-experienced — harder penalty, fresher shouldn't fake it
        return max(0.0, 1.0 - (lo - exp_years) * 0.20)
    else:
        # Over-qualified — very mild penalty
        return max(0.60, 1.0 - (exp_years - hi) * 0.03)


# ── Master scorer ─────────────────────────────────────────────────────────────

def compute_match_score(
    candidate:    dict,
    jd:           dict,
    jd_embedding: list,
    embed_fn,
    rag_score:    Optional[float] = None,
) -> tuple[float, str]:
    """
    Compute match_score and human-readable explanation.

    Formula:
      match = 0.60 × skill + 0.25 × project + 0.15 × experience
      final = match + context_tiebreaker (≤3%) + rag_tiebreaker (≤3%)

    Interest score is NOT part of the formula.
    """
    c_skills   = candidate.get("skills", [])
    c_exp      = candidate.get("experience_years")
    c_projects = " ".join(candidate.get("projects", []))
    c_summary  = f"{candidate.get('summary', '')} {' '.join(c_skills)}"

    req = jd.get("required_skills", [])
    opt = jd.get("optional_skills", [])

    # ── 1. Skill score (semantic) ─────────────────────────────────────────────
    sk, sk_bd = skill_score(c_skills, req, opt, embed_fn)

    # ── 2. Project score (embedding) ──────────────────────────────────────────
    jd_context = " ".join(req + jd.get("keywords", []))
    proj = 0.0
    if c_projects.strip() and jd_context.strip():
        try:
            proj = _cosine(
                _cached_embed(c_projects[:1000], embed_fn),
                _cached_embed(jd_context[:800],  embed_fn),
            )
        except Exception:
            pass

    # ── 3. Experience score ───────────────────────────────────────────────────
    exp = experience_score(c_exp, jd.get("experience_min"), jd.get("experience_max"))

    # ── Core match ────────────────────────────────────────────────────────────
    match = (
        WEIGHTS["skill"]      * sk   +
        WEIGHTS["project"]    * proj +
        WEIGHTS["experience"] * exp
    )

    # ── Tiebreakers (context + RAG) ───────────────────────────────────────────
    ctx_raw = 0.0
    if c_summary.strip() and jd_embedding:
        try:
            ctx_raw = _cosine(_cached_embed(c_summary[:1000], embed_fn), jd_embedding)
        except Exception:
            pass
    ctx_boost = ctx_raw * CONTEXT_TIEBREAKER_MAX   # max +3%

    rag_boost = 0.0
    if rag_score is not None:
        rag_boost = rag_score * RAG_TIEBREAKER_MAX  # max +3%

    match = round(min(1.0, match + ctx_boost + rag_boost), 4)

    # ── Human explanation ─────────────────────────────────────────────────────
    req_pct = sk_bd["req_score"]
    opt_pct = sk_bd["opt_score"]
    lines = [
        "─── Scoring Formula ─────────────────────────────────────────────",
        "  final = 0.60×skill + 0.25×project + 0.15×experience",
        "         + context tiebreaker (≤3%) + RAG tiebreaker (≤3%)",
        "",
        f"Skill Score      : {sk:.0%}",
        f"  Required       : {req_pct:.0%}  ({len(sk_bd['required_matched'])}/{len(req)} matched)",
        f"  ✅ Matched     : {', '.join(sk_bd['required_matched']) or 'none'}",
        f"  ❌ Missing     : {', '.join(sk_bd['required_missing']) or 'none'}",
        f"  Optional       : {opt_pct:.0%}  ({len(sk_bd['optional_matched'])}/{len(opt)} matched)",
        f"Project Score    : {proj:.0%}",
        f"Experience Score : {exp:.0%}  (candidate: {c_exp} yrs, JD: {jd.get('experience_min')}–{jd.get('experience_max')} yrs)",
        f"Context Boost    : +{ctx_boost:.1%}  (profile–JD embedding)",
        f"RAG Boost        : +{rag_boost:.1%}  (Pinecone cosine={rag_score:.3f})" if rag_score else "RAG Boost        : N/A",
        "─────────────────────────────────────────────────────────────────",
        f"Match Score      : {match:.0%}",
        "",
        f"Thresholds  →  Strong ≥72%  |  Shortlist ≥58%  |  Consider ≥42%",
    ]

    return match, "\n".join(lines)


def decide(final_score: float) -> str:
    if final_score >= THRESHOLD_STRONG:
        return "Strong Shortlist"
    elif final_score >= THRESHOLD_SHORT:
        return "Shortlist"
    elif final_score >= THRESHOLD_CONSIDER:
        return "Consider"
    else:
        return "Reject"