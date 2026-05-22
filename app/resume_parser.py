"""
resume_parser.py — Robust resume parser.

Features:
  - Retry up to 3 times on parse failure (LLM or JSON)
  - Quarantines failed resumes to <resume_dir>/../resumes_failed/
  - After primary pass, re-attempts quarantined files once more
  - Normalises output regardless of LLM response format
"""

import os
import re
import json
import shutil
import logging
import time
from pathlib import Path
from typing import Optional

import fitz          # PyMuPDF
import docx

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 1.5   # seconds between retries

# ── Text extraction ──────────────────────────────────────────────────────────

def extract_text_from_pdf(path: str) -> str:
    try:
        doc = fitz.open(path)
        return "\n".join(page.get_text() for page in doc).strip()
    except Exception as e:
        logger.error("PDF extraction failed for %s: %s", path, e)
        return ""


def extract_text_from_docx(path: str) -> str:
    try:
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs).strip()
    except Exception as e:
        logger.error("DOCX extraction failed for %s: %s", path, e)
        return ""


def extract_text(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext == ".docx":
        return extract_text_from_docx(path)
    elif ext == ".txt":
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    else:
        logger.warning("Unsupported file type: %s", ext)
        return ""

# ── Prompt ───────────────────────────────────────────────────────────────────

RESUME_PROMPT = """You are a precise JSON extractor for resumes/CVs.

Extract the following fields from the resume text below.
Return ONLY valid JSON — no markdown, no backticks, no comments, no trailing commas.

Schema:
{{
  "name": "Full Name",
  "email": "email or null",
  "phone": "phone or null",
  "experience_years": <number or null>,
  "skills": ["exhaustive list of ALL technical skills mentioned anywhere in the resume"],
  "projects": ["short description of each project"],
  "education": "highest degree and institution",
  "summary": "2-3 sentence professional summary"
}}

Rules:
- experience_years: ONLY count actual paid work experience (internships, jobs, freelance).
  DO NOT count education years, graduation years, or project duration.
  If the resume has no work experience section, return null.
  If there are internships with clear durations, sum those months and convert to years.
- skills: Extract ALL skills from EVERY section — including skills listed under "Bonus",
  "Learning", "Interests", or embedded in project descriptions.
  Examples: if projects mention "built with Gemini LLM" → add "Gemini".
  If interests mention "Generative AI" → add "Generative AI".
  If a section says "AI/LLM: Prompt Engineering, LangChain, RAG" → add all of them.
  Include both the full name AND common abbreviations when both appear
  (e.g. add both "Retrieval-Augmented Generation" and "RAG").
- projects: one string per project (title + one-line description)
- Return ONLY the JSON object, nothing else

Resume:
{resume_text}
"""

# ── JSON cleaning (same as jd_parser) ────────────────────────────────────────

def _clean_json(raw: str) -> str:
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().strip("`").strip()
    raw = re.sub(r"//[^\n]*", "", raw)
    raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.DOTALL)
    raw = re.sub(r",\s*([\}\]])", r"\1", raw)
    return raw.strip()


def _normalize_list(lst) -> list:
    if not isinstance(lst, list):
        return []
    return [str(i).strip() for i in lst if str(i).strip()]


def _safe_float(val) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

# ── Regex fallback ───────────────────────────────────────────────────────────

_SKILL_RE = re.compile(
    r"\b(Python|Java(?:Script|Script)?|TypeScript|C\+\+|C#|Go|Rust|SQL|NoSQL|"
    r"MongoDB|PostgreSQL|MySQL|Redis|AWS|Azure|GCP|Docker|Kubernetes|"
    r"FastAPI|Flask|Django|React|Angular|Vue|Machine Learning|Deep Learning|"
    r"NLP|Computer Vision|TensorFlow|PyTorch|Scikit-?learn|Keras|Pandas|NumPy|"
    r"Spark|Kafka|Airflow|Generative AI|LLM|MLOps|OpenAI|Gemini|LangChain|"
    r"RAG|Pinecone|Git|Linux|Bash|Tableau|Power BI|Excel|Communication|Agile)\b",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_NAME_RE  = re.compile(r"^([A-Z][a-z]+(?: [A-Z][a-z]+)+)", re.MULTILINE)
_EXP_RE   = re.compile(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience)?", re.IGNORECASE)


def _fallback_parse(text: str, filename: str) -> dict:
    name_m  = _NAME_RE.search(text)
    email_m = _EMAIL_RE.search(text)
    exp_m   = _EXP_RE.search(text)
    skills  = list(dict.fromkeys(m.group(0).title() for m in _SKILL_RE.finditer(text)))

    # Estimate name from filename if not found
    name = name_m.group(1) if name_m else Path(filename).stem.replace("_", " ").title()

    return {
        "name": name,
        "email": email_m.group(0) if email_m else None,
        "phone": None,
        "experience_years": _safe_float(exp_m.group(1)) if exp_m else None,
        "skills": skills,
        "projects": [],
        "education": "",
        "summary": f"Resume parsed via fallback extractor from {filename}.",
        "_source_file": filename,
    }

# ── Single-file LLM parse (with retries) ─────────────────────────────────────

def _llm_parse_with_retry(text: str, filename: str, model) -> Optional[dict]:
    """Attempt LLM parse up to MAX_RETRIES times. Returns dict or None."""
    prompt = RESUME_PROMPT.format(resume_text=text[:6000])  # truncate for token safety

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = model.generate_content(prompt)
            raw = response.text.strip()
            cleaned = _clean_json(raw)
            parsed = json.loads(cleaned)

            return {
                "name": str(parsed.get("name") or Path(filename).stem).strip(),
                "email": parsed.get("email"),
                "phone": parsed.get("phone"),
                "experience_years": _safe_float(parsed.get("experience_years")),
                "skills": _normalize_list(parsed.get("skills", [])),
                "projects": _normalize_list(parsed.get("projects", [])),
                "education": str(parsed.get("education") or ""),
                "summary": str(parsed.get("summary") or ""),
                "raw_text": text,
                "_source_file": filename,
                "_parse_method": "llm",
            }

        except json.JSONDecodeError as e:
            logger.warning(
                "⚠️  [%s] Attempt %d/%d — JSON parse failed: %s",
                filename, attempt, MAX_RETRIES, e,
            )
        except Exception as e:
            logger.warning(
                "⚠️  [%s] Attempt %d/%d — LLM error: %s",
                filename, attempt, MAX_RETRIES, e,
            )

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    return None  # all retries exhausted

# ── Quarantine helpers ────────────────────────────────────────────────────────

def _quarantine(filepath: str, failed_dir: str) -> str:
    os.makedirs(failed_dir, exist_ok=True)
    dest = os.path.join(failed_dir, os.path.basename(filepath))
    shutil.copy2(filepath, dest)
    logger.warning("📦  Quarantined: %s  →  %s", filepath, dest)
    return dest

# ── Public API ────────────────────────────────────────────────────────────────

def parse_resumes(resume_dir: str, model, cache: dict) -> list:
    """
    Parse all resumes in resume_dir.
    On failure after MAX_RETRIES, quarantine to resumes_failed/ and
    re-attempt quarantined files at the end.

    Returns list of parsed candidate dicts.
    """
    resume_dir = os.path.abspath(resume_dir)
    failed_dir = os.path.join(os.path.dirname(resume_dir), "resumes_failed")
    supported   = {".pdf", ".docx", ".txt"}

    files = [
        os.path.join(resume_dir, f)
        for f in os.listdir(resume_dir)
        if Path(f).suffix.lower() in supported
    ]

    results = []
    quarantined_paths = []

    # ── Primary pass ─────────────────────────────────────────────────────────
    for filepath in files:
        filename = os.path.basename(filepath)

        # Cache hit
        if filename in cache:
            logger.info("✅  Cache hit: %s", filename)
            results.append(cache[filename])
            continue

        logger.info("🔄  Parsing: %s", filename)
        text = extract_text(filepath)

        if not text.strip():
            logger.warning("⚠️  Empty text for %s — quarantining", filename)
            quarantined_paths.append(_quarantine(filepath, failed_dir))
            continue

        parsed = _llm_parse_with_retry(text, filename, model)

        if parsed is None:
            logger.warning("❌  All retries failed for %s — using fallback + quarantine", filename)
            parsed = _fallback_parse(text, filename)
            parsed["_parse_method"] = "fallback_quarantined"
            quarantined_paths.append(_quarantine(filepath, failed_dir))

        cache[filename] = parsed
        results.append(parsed)

    # ── Quarantine re-attempt pass ────────────────────────────────────────────
    if quarantined_paths:
        logger.info("🔁  Re-attempting %d quarantined resume(s)...", len(quarantined_paths))
        for filepath in quarantined_paths:
            filename = os.path.basename(filepath)
            text = extract_text(filepath)
            if not text.strip():
                continue

            parsed = _llm_parse_with_retry(text, filename, model)
            if parsed:
                logger.info("✅  Quarantine recovery succeeded: %s", filename)
                parsed["_parse_method"] = "llm_recovered"
                # Update the result already added (it may have been added as fallback)
                for i, r in enumerate(results):
                    if r.get("_source_file") == filename or r.get("name", "").lower() in filename.lower():
                        results[i] = parsed
                        break
                cache[filename] = parsed

    return results