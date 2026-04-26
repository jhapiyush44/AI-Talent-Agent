from sentence_transformers import SentenceTransformer, util
import numpy as np
from difflib import SequenceMatcher

# ---------- LOAD MODEL ----------
model = SentenceTransformer('all-MiniLM-L6-v2')


# ---------- NORMALIZATION ----------
def normalize(text):
    return text.lower().strip()


# ---------- SYNONYMS ----------
SYNONYMS = {
    "machine learning": ["ml", "ai"],
    "deep learning": ["dl", "neural networks"],
    "fastapi": ["api"],
    "sql": ["database", "db"],
    "javascript": ["js"],
    "react": ["reactjs"],
    "node.js": ["node"],
}


# ---------- KEYWORD MATCH ----------
def keyword_match(skill, candidate_skills):
    skill = normalize(skill)

    for cs in candidate_skills:
        cs = normalize(cs)

        if skill == cs:
            return True

        if skill in cs or cs in skill:
            return True

        for key, values in SYNONYMS.items():
            if skill == key and any(v in cs for v in values):
                return True

        if SequenceMatcher(None, skill, cs).ratio() > 0.8:
            return True

    return False


# ---------- EMBEDDING CACHE ----------
def embed(texts, cache):
    results = []

    for t in texts:
        if t not in cache:
            cache[t] = model.encode(t, convert_to_tensor=True)
        results.append(cache[t])

    return results


# ---------- SKILL SCORE ----------
def skill_score(candidate, jd):
    candidate_skills = candidate.get("skills", [])
    required = jd.get("required_skills", [])
    optional = jd.get("optional_skills", [])

    if not required:
        return 0

    req_matches = 0
    critical_missing = 0

    for skill in required:
        if keyword_match(skill, candidate_skills):
            req_matches += 1
        else:
            if skill.lower() in ["python", "machine learning", "sql"]:
                critical_missing += 1

    req_score = req_matches / len(required)

    opt_matches = sum(
        1 for skill in optional
        if keyword_match(skill, candidate_skills)
    )

    opt_score = opt_matches / max(len(optional), 1)

    penalty = 0.1 * critical_missing

    final = req_score + (0.2 * opt_score) - penalty

    return max(0, min(final, 1))


# ---------- PROJECT SCORE ----------
def project_score(candidate, jd, cache):
    projects = candidate.get("projects", [])
    jd_text = jd.get("jd_summary", "")

    if not projects or not jd_text:
        return 0

    proj_texts = []
    for p in projects:
        if isinstance(p, dict):
            proj_texts.append(p.get("title", "") + " " + p.get("description", ""))
        else:
            proj_texts.append(str(p))

    proj_emb = embed(proj_texts, cache)
    jd_emb = embed([jd_text], cache)[0]

    sims = [util.cos_sim(p, jd_emb).item() for p in proj_emb]

    return max(sims)


# ---------- CONTEXT SCORE ----------
def context_score(candidate, jd, cache):
    text = (
        candidate.get("bio", "") + " " +
        " ".join(candidate.get("skills", []))
    )

    jd_summary = jd.get("jd_summary", "")

    if not text or not jd_summary:
        return 0

    text_emb = embed([text], cache)[0]
    jd_emb = embed([jd_summary], cache)[0]

    return util.cos_sim(text_emb, jd_emb).item()


# ---------- EXPERIENCE ----------
def experience_score(candidate, jd):
    exp = candidate.get("experience", 0)

    min_exp = jd.get("experience_min", 0)
    max_exp = jd.get("experience_max", 5)

    if min_exp <= exp <= max_exp:
        return 1
    elif exp < min_exp:
        return exp / max(min_exp, 1)
    else:
        return 0.8


# ---------- FINAL ----------
def compute_match(candidate, jd):
    cache = {}

    s = skill_score(candidate, jd)
    e = experience_score(candidate, jd)
    p = project_score(candidate, jd, cache)
    c = context_score(candidate, jd, cache)

    final_score = (
        0.5 * s +
        0.2 * p +
        0.2 * e +
        0.1 * c
    )

    explanation = f"""
Skill Score: {round(s, 2)}
Project Score: {round(p, 2)}
Experience Score: {round(e, 2)}
Context Score: {round(c, 2)}
""".strip()

    return final_score, explanation