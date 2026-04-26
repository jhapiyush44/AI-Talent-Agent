import google.generativeai as genai
import os
import json
import re
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ---------- NORMALIZATION ----------
def normalize_skill(skill):
    return skill.strip().title()


# ---------- EMAIL ----------
def extract_email(text):
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return match.group() if match else "Not Provided"


# ---------- NAME ----------
def extract_name(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    for line in lines[:5]:
        if any(k in line.lower() for k in ["email", "phone", "@", "contact"]):
            continue

        words = line.split()
        if 2 <= len(words) <= 4 and all(w.isalpha() for w in words):
            return line

    return "Unknown Candidate"


# ---------- SAFE JSON ----------
def safe_parse_json(content):
    content = content.replace("```json", "").replace("```", "")

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group())
    except:
        return None


# ---------- FALLBACK ----------
def fallback_parser(text):
    text_lower = text.lower()

    name = extract_name(text)
    email = extract_email(text)

    skill_keywords = [
        "python", "java", "sql", "aws", "docker", "react", "node",
        "machine learning", "deep learning", "fastapi",
        "accounting", "tally", "gst", "taxation", "audit",
        "financial analysis", "bookkeeping",
        "recruitment", "talent acquisition", "payroll",
        "performance management", "training",
        "excel", "power bi", "communication", "leadership"
    ]

    skills = []
    for skill in skill_keywords:
        if skill in text_lower:
            skills.append(normalize_skill(skill))

    exp_match = re.search(r"(\d+)\s*\+?\s*(years|yrs)", text_lower)
    experience = int(exp_match.group(1)) if exp_match else 0

    projects = []
    for line in text.split("\n"):
        if "project" in line.lower() and len(line.strip()) > 5:
            projects.append({
                "title": line.strip(),
                "description": ""
            })

    if not projects:
        projects = [{
            "title": "General Experience",
            "description": text[:200]
        }]

    return {
        "name": name,
        "email": email,
        "skills": list(set(skills)),
        "experience": experience,
        "projects": projects,
        "bio": text[:300]
    }


# ---------- MAIN ----------
def parse_resume(text):

    prompt = f"""
    You are an expert resume parser.

    Extract structured candidate information from the resume.

    STRICT RULES:
    - Return ONLY valid JSON (no markdown, no ```json, no explanation)
    - Do NOT add extra text before or after JSON
    - Do NOT hallucinate missing data
    - If a field is not found, return empty string or empty list

    EXTRACTION RULES:

    1. NAME
    - Extract full candidate name (usually at top)
    - Avoid picking email, phone, or headings

    2. EMAIL
    - Extract email address exactly

    3. SKILLS
    - Extract ALL relevant skills
    - Include technical, professional, and domain-specific skills
    - Examples:
      - Tech: Python, Docker, SQL
      - Finance: Accounting, GST, Audit
      - HR: Recruitment, Payroll
    - Normalize skill names (e.g., "ml" → "Machine Learning")

    4. EXPERIENCE
    - Extract TOTAL years of experience as integer
    - If unclear, estimate conservatively

    5. PROJECTS
    - Extract 1–5 most important projects
    - Each project must include:
      - title (short)
      - description (1 line)

    6. BIO
    - Short 2–3 line professional summary

    OUTPUT FORMAT:

    {{
      "name": "",
      "email": "",
      "skills": [],
      "experience": 0,
      "projects": [
        {{
          "title": "",
          "description": ""
        }}
      ],
      "bio": ""
    }}

    Resume:
    {text[:4000]}
    """

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        parsed = safe_parse_json(response.text)

        if not parsed:
            print("⚠️ LLM failed → fallback parser")
            return fallback_parser(text)

        parsed["name"] = parsed.get("name") or extract_name(text)
        parsed["email"] = parsed.get("email") or extract_email(text)
        parsed["skills"] = [normalize_skill(s) for s in parsed.get("skills", [])]
        parsed["experience"] = parsed.get("experience", 0)
        parsed["projects"] = parsed.get("projects", [])
        parsed["bio"] = parsed.get("bio") or text[:300]

        cleaned_projects = []
        for proj in parsed["projects"]:
            if isinstance(proj, dict):
                cleaned_projects.append({
                    "title": proj.get("title", "Project"),
                    "description": proj.get("description", "")
                })
            else:
                cleaned_projects.append({
                    "title": "Project",
                    "description": str(proj)
                })

        if not cleaned_projects:
            cleaned_projects = [{
                "title": "General Experience",
                "description": text[:200]
            }]

        parsed["projects"] = cleaned_projects

        return parsed

    except Exception as e:
        print("LLM Error → fallback:", e)
        return fallback_parser(text)