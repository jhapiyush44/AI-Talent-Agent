from fastapi import FastAPI, UploadFile, File, Form
import json
import os

from app.agent import run_agent
from app.utils import (
    extract_text_from_pdf,
    extract_text_from_docx,
    load_candidates_from_resumes
)

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESUME_FOLDER = os.path.join(BASE_DIR, "..", "resumes")
FALLBACK_JSON = os.path.join(BASE_DIR, "data", "candidates.json")


@app.get("/health")
async def health_check():
    """Health check endpoint for frontend to verify backend is running"""
    return {"status": "ok", "message": "AI Talent Agent backend is running"}


@app.post("/run-agent")
async def run_agent_api(
    jd_file: UploadFile = File(None),
    jd_text: str = Form(None),
    top_k: int = Form(5),
):
    # ---------- INPUT VALIDATION ----------
    top_k = max(1, min(top_k, 50))

    jd_content = ""

    # ---------- STEP 1: JD EXTRACTION ----------
    try:
        if jd_file:
            filename = jd_file.filename.lower()

            jd_file.file.seek(0)  # 🔥 FIX

            if filename.endswith(".pdf"):
                jd_content = extract_text_from_pdf(jd_file.file)

            elif filename.endswith(".docx"):
                jd_content = extract_text_from_docx(jd_file.file)

            else:
                return {"error": "Unsupported JD file format"}

        elif jd_text:
            jd_content = jd_text.strip()

    except Exception as e:
        print("❌ JD extraction error:", str(e))
        return {"error": "Failed to process JD file"}

    if not jd_content:
        return {"error": "Provide JD text or upload JD file"}

    # ---------- STEP 2: LOAD CANDIDATES ----------
    try:
        print("📂 Loading resumes from:", RESUME_FOLDER)
        candidates = load_candidates_from_resumes(RESUME_FOLDER)

    except Exception as e:
        print("❌ Resume loading error:", str(e))
        candidates = []

    # ---------- FALLBACK ----------
    if not candidates:
        print("⚠️ Fallback: using candidates.json")

        try:
            with open(FALLBACK_JSON) as f:
                candidates = json.load(f)
        except Exception as e:
            print("❌ Fallback failed:", str(e))
            return {"error": "No candidates available"}

    # ---------- STEP 3: RUN AGENT ----------
    try:
        result = run_agent(jd_content, candidates, top_k=top_k)
        return result

    except Exception as e:
        print("❌ Agent execution error:", str(e))
        return {"error": "Failed to process candidates"}