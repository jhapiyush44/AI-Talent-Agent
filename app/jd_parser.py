from google import genai
import os
import json
import re
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# ---------- NORMALIZATION ----------
def normalize_skill(skill):
    return skill.strip().title()


def normalize_skills(skills):
    mapping = {
        # ML / AI
        "ml": "Machine Learning",
        "machine learning": "Machine Learning",
        "machine-learning": "Machine Learning",
        "ai": "Machine Learning",

        # Backend
        "fast api": "FastAPI",
        "fastapi": "FastAPI",
        "rest api": "API",
        "rest apis": "API",
        "restful api": "API",

        # Frontend / JS
        "react.js": "React",
        "reactjs": "React",
        "nodejs": "Node.js",
        "node.js": "Node.js",
        "node js": "Node.js",

        # Tools
        "github": "Git",
        "gitlab": "Git",
        "aws cloud": "AWS"
    }

    normalized = []
    for s in skills:
        key = s.lower().strip()
        mapped = mapping.get(key, s)
        normalized.append(normalize_skill(mapped))

    return list(set(normalized))


# ---------- SAFE JSON PARSER (ROBUST) ----------
def safe_parse_json(content):
    # remove markdown
    content = content.replace("```json", "").replace("```", "")

    # remove non-ascii garbage (🔥 fixes your bug)
    content = re.sub(r"[^\x00-\x7F]+", "", content)

    # extract JSON block
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return None

    json_str = match.group()

    # remove trailing commas
    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = re.sub(r",\s*]", "]", json_str)

    try:
        return json.loads(json_str)
    except Exception as e:
        print("❌ JSON parse failed:", e)
        return None


# ---------- DOMAIN-AGNOSTIC FALLBACK ----------
def extract_skills_from_text(text):
    text = text.lower()

    skill_bank = [
        # Tech
        "python", "machine learning", "sql", "fastapi", "flask",
        "pandas", "numpy", "scikit-learn", "docker", "aws", "gcp",
        "deep learning", "nlp", "computer vision",

        # Data
        "excel", "power bi", "tableau", "data analysis",
        "data visualization", "statistics",

        # Finance
        "accounting", "tally", "gst", "taxation", "audit",
        "financial analysis", "bookkeeping",

        # HR / Business
        "recruitment", "talent acquisition", "payroll",
        "performance management", "training",
        "communication", "leadership",

        # General
        "reporting", "presentation", "ms office"
    ]

    found = []

    for skill in skill_bank:
        if skill in text:
            found.append(skill.title())

    found = list(set(found))

    return found[:7]  # limit


# ---------- MAIN ----------
def parse_jd(jd_text):
    prompt = f"""
    Extract structured hiring requirements from this job description.

    Rules:
    - Identify REQUIRED (must-have) skills (core skills only)
    - Identify OPTIONAL (good-to-have) skills
    - REQUIRED skills should be the most important 4–8 skills
    - OPTIONAL skills can include secondary tools, frameworks, or domains
    - Extract technologies, frameworks, languages, tools
    - Normalize abbreviations (ML → Machine Learning)
    - Avoid duplicates
    - Keep keywords short (1–2 words)

    Return ONLY valid JSON:
    {{
      "required_skills": [],
      "optional_skills": [],
      "experience_min": int,
      "experience_max": int,
      "keywords": []
    }}

    Job Description:
    {jd_text[:4000]}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        content = response.text
        print("RAW LLM RESPONSE:\n", content)

        jd = safe_parse_json(content)

        # ---------- 🔥 FALLBACK IF LLM FAILS ----------
        if not jd:
            print("⚠️ LLM failed → using fallback extractor")

            extracted = extract_skills_from_text(jd_text)

            jd = {
                "required_skills": extracted[:5],
                "optional_skills": extracted[5:],
                "experience_min": 0,
                "experience_max": 5,
                "keywords": extracted
            }

    except Exception as e:
        print("❌ JD Parsing Error:", e)

        extracted = extract_skills_from_text(jd_text)

        jd = {
            "required_skills": extracted[:5],
            "optional_skills": extracted[5:],
            "experience_min": 0,
            "experience_max": 5,
            "keywords": extracted
        }

    # ---------- NORMALIZE ----------
    jd["required_skills"] = normalize_skills(jd.get("required_skills", []))
    jd["optional_skills"] = normalize_skills(jd.get("optional_skills", []))

    # remove overlap
    jd["optional_skills"] = [
        s for s in jd["optional_skills"]
        if s not in jd["required_skills"]
    ]

    # keywords → keep raw for embeddings
    jd["keywords"] = [k.lower().strip() for k in jd.get("keywords", [])]

    # experience defaults
    jd["experience_min"] = jd.get("experience_min", 0)
    jd["experience_max"] = jd.get("experience_max", 5)

    # fallback if still empty
    if not jd["required_skills"]:
        extracted = extract_skills_from_text(jd_text)
        jd["required_skills"] = extracted[:3]

    # ---------- 🔥 EMBEDDING CONTEXT ----------
    jd["jd_summary"] = " ".join(
        jd["required_skills"] +
        jd["optional_skills"] +
        jd["keywords"]
    )

    # ---------- DEBUG ----------
    print("JD REQUIRED:", jd["required_skills"])
    print("JD OPTIONAL:", jd["optional_skills"])

    return jd