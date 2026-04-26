from google import genai
import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# 🔥 Safe JSON parsing (robust)
def safe_parse_json(content):
    try:
        return json.loads(content)
    except:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass

        return {
            "response": "Not actively looking for a change right now.",
            "interest_score": 0.5,
            "reason": "Fallback parsing"
        }


def simulate_interest(candidate, jd):
    prompt = f"""
    You are simulating a job candidate's realistic response.

    Candidate Profile:
    Skills: {candidate.get("skills", [])}
    Experience: {candidate.get("experience", 0)} years
    Projects: {candidate.get("projects", [])}
    Bio: {candidate.get("bio", "")}

    Job Requirements:
    Required Skills: {jd.get("required_skills", [])}
    Optional Skills: {jd.get("optional_skills", [])}
    Experience: {jd.get("experience_min", 0)}-{jd.get("experience_max", 5)}

    Instructions:
    - Be realistic, not always positive
    - If strong skill match → high interest (0.7–1.0)
    - If partial match → moderate interest (0.4–0.7)
    - If poor match → low interest (0.0–0.4)
    - Keep response short (2–3 lines)
    - Do NOT hallucinate experience or skills

    Respond to:
    "Hi, are you interested in this role?"

    Then rate interest from 0 to 1.

    Return ONLY valid JSON:
    {{
      "response": "...",
      "interest_score": float,
      "reason": "short reason"
    }}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        content = response.text

        result = safe_parse_json(content)

        # 🔥 Ensure structure exists
        if not isinstance(result, dict):
            result = {}

        result.setdefault("response", "Not actively looking for a change right now.")
        result.setdefault("interest_score", 0.5)
        result.setdefault("reason", "Default fallback")

        # 🔥 Clamp score (important)
        try:
            score = float(result.get("interest_score", 0.5))
        except:
            score = 0.5

        result["interest_score"] = max(0, min(score, 1))

        return result

    except Exception as e:
        print("Simulator Error:", e)

        return {
            "response": "Not actively looking for a change right now.",
            "interest_score": 0.5,
            "reason": "Fallback due to API issue"
        }