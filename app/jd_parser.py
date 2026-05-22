"""
jd_parser.py — Robust JD parser with LLM + regex fallback.
Fixes:
  - Strips markdown fences and invalid JSON (like inline comments) before parsing
  - Sanitizes keys to Title-Case strings
  - Validates required fields; falls back gracefully
"""

import re
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Prompt ──────────────────────────────────────────────────────────────────

JD_PROMPT = """You are a precise JSON extractor for job descriptions.

Extract the following fields from the job description below.
Return ONLY valid JSON — no markdown, no backticks, no comments, no trailing commas.

Schema:
{{
  "required_skills": ["list of must-have technical skills"],
  "optional_skills": ["list of nice-to-have skills"],
  "experience_min": <number or null>,
  "experience_max": <number or null>,
  "keywords": ["important domain keywords"]
}}

Rules:
- Skill names must be SHORT and BARE — the technology name only, without action verbs or qualifiers.
  WRONG: "Generative AI Development", "AI Model Optimization", "API Development"
  RIGHT: "Generative AI", "Model Optimization", "FastAPI" or just "APIs"
  WRONG: "Large Language Models (LLMs)" — RIGHT: "Large Language Models"
  WRONG: "ML Ops" — RIGHT: "MLOps"
  This is critical: skills will be matched against resumes, so they must match how engineers write them.
- Skill names must be plain strings (no inline notes or sub-objects)
- experience_min / experience_max must be numbers (years) or null
- Do NOT add any text outside the JSON object

Job Description:
{jd_text}
"""

# ── Helpers ──────────────────────────────────────────────────────────────────

def _clean_json_string(raw: str) -> str:
    """Strip markdown fences, JS-style comments, trailing commas."""
    # Remove ```json ... ``` or ``` ... ``` fences
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = raw.strip().strip("`").strip()

    # Remove JS-style // single-line comments (common LLM mistake)
    raw = re.sub(r"//[^\n]*", "", raw)

    # Remove JS-style /* */ block comments
    raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.DOTALL)

    # Remove trailing commas before } or ]
    raw = re.sub(r",\s*([\}\]])", r"\1", raw)

    return raw.strip()


def _normalize_skill_list(lst) -> list:
    """Ensure a list contains only plain strings."""
    if not isinstance(lst, list):
        return []
    cleaned = []
    for item in lst:
        if isinstance(item, str) and item.strip():
            cleaned.append(item.strip())
        elif isinstance(item, dict):
            cleaned.extend(k.strip() for k in item.keys() if k.strip())
    return cleaned


def _to_title_case_list(lst: list) -> list:
    # Do NOT apply .title() — it mangles acronyms like GCP→Gcp, LLMs→Llms,
    # PyTorch→Pytorch, MLOps→Mlops. Return skills exactly as the LLM wrote them.
    return lst


def _safe_float(val) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

# ── Regex-based fallback ─────────────────────────────────────────────────────

_SKILL_KEYWORDS = [
    "Python", "Java", "JavaScript", "TypeScript", "C\\+\\+", "C#", "Go", "Rust",
    "SQL", "NoSQL", "MongoDB", "PostgreSQL", "MySQL", "Redis",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform",
    "FastAPI", "Flask", "Django", "Spring", "React", "Angular", "Vue",
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "TensorFlow", "PyTorch", "Scikit.?learn", "Keras",
    "Pandas", "NumPy", "Spark", "Kafka", "Airflow",
    "Generative AI", "LLM", "Large Language Models", "MLOps",
    "OpenAI", "Gemini", "Claude", "LangChain", "RAG",
    "Communication", "Excel", "Agile", "Scrum",
]

def _fallback_extract(jd_text: str) -> dict:
    """Pure-regex fallback — extracts skills when LLM JSON is unparseable."""
    found = []
    for kw in _SKILL_KEYWORDS:
        if re.search(kw, jd_text, re.IGNORECASE):
            found.append(re.sub(r"\\", "", kw).title())

    # Experience range
    exp_match = re.search(
        r"(\d+(?:\.\d+)?)\s*[-–to]+\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)",
        jd_text, re.IGNORECASE,
    )
    exp_min = float(exp_match.group(1)) if exp_match else None
    exp_max = float(exp_match.group(2)) if exp_match else None

    if not exp_match:
        single = re.search(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)", jd_text, re.IGNORECASE)
        if single:
            exp_min = float(single.group(1))

    split = len(found) // 2
    return {
        "required_skills": found[:split] or found,
        "optional_skills": found[split:],
        "experience_min": exp_min,
        "experience_max": exp_max,
        "keywords": found[:10],
    }

# ── Main parser ──────────────────────────────────────────────────────────────

def parse_jd(jd_text: str, model) -> dict:
    """Parse a job description; returns a normalised dict."""
    prompt = JD_PROMPT.format(jd_text=jd_text)

    raw = ""
    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
        logger.info("RAW LLM JD RESPONSE:\n%s", raw)

        cleaned = _clean_json_string(raw)
        parsed = json.loads(cleaned)

        result = {
            "required_skills": _to_title_case_list(
                _normalize_skill_list(parsed.get("required_skills", []))
            ),
            "optional_skills": _to_title_case_list(
                _normalize_skill_list(parsed.get("optional_skills", []))
            ),
            "experience_min": _safe_float(parsed.get("experience_min")),
            "experience_max": _safe_float(parsed.get("experience_max")),
            "keywords": _normalize_skill_list(parsed.get("keywords", [])),
        }

        logger.info(
            "JD REQUIRED: %s | OPTIONAL: %s",
            result["required_skills"],
            result["optional_skills"],
        )
        return result

    except json.JSONDecodeError as e:
        logger.warning("❌ JSON parse failed: %s  →  using fallback extractor", e)
        logger.debug("Raw text was:\n%s", raw)
        fallback = _fallback_extract(jd_text)
        logger.info(
            "⚠️  FALLBACK JD REQUIRED: %s | OPTIONAL: %s",
            fallback["required_skills"],
            fallback["optional_skills"],
        )
        return fallback

    except Exception as e:
        logger.error("❌ JD parse error: %s  →  using fallback extractor", e)
        return _fallback_extract(jd_text)