"""
main.py — FastAPI backend for AI Talent Agent.
"""

import os
import io
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()

import fitz
import docx as _docx

from .agent import run_agent


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Talent Agent API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RESUME_DIR = os.getenv("RESUME_DIR", os.path.join(os.path.dirname(__file__), "..", "resumes"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_text_from_upload(upload: UploadFile) -> str:
    data = upload.file.read()
    name = upload.filename.lower()
    if name.endswith(".pdf"):
        doc = fitz.open(stream=data, filetype="pdf")
        return "\n".join(p.get_text() for p in doc).strip()
    elif name.endswith(".docx"):
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        doc = _docx.Document(tmp_path)
        os.unlink(tmp_path)
        return "\n".join(p.text for p in doc.paragraphs).strip()
    elif name.endswith(".txt"):
        return data.decode("utf-8", errors="replace").strip()
    return ""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/run-agent")
async def run_agent_endpoint(
    jd_text: str   = Form(default=""),
    top_k:   int   = Form(default=10),
    jd_file: UploadFile = File(default=None),
):
    # Resolve JD text
    final_jd = jd_text.strip()
    if jd_file and jd_file.filename:
        extracted = _extract_text_from_upload(jd_file)
        if extracted:
            final_jd = extracted

    if not final_jd:
        raise HTTPException(status_code=400, detail="No job description provided.")

    try:
        result = run_agent(
            jd_text=final_jd,
            resume_dir=RESUME_DIR,
            top_k=top_k,
        )
        return result
    except Exception as e:
        logger.exception("Agent execution error")
        return {"error": str(e), "top_candidates": []}