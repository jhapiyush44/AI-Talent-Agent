# 🎯 AI Talent Agent v2.0

> **Intelligent candidate screening powered by Gemini LLM · SentenceTransformers · Pinecone RAG**

An end-to-end AI recruitment pipeline that automatically parses job descriptions, extracts structured data from resumes, scores candidates through a hybrid semantic scoring engine, and ranks them for recruiter review — all through a polished Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Gemini](https://img.shields.io/badge/LLM-Gemini_API-4285F4?style=flat&logo=google&logoColor=white)](https://aistudio.google.com)
[![SentenceTransformers](https://img.shields.io/badge/Embeddings-SentenceTransformers-orange?style=flat)](https://sbert.net)
[![VectorDB](https://img.shields.io/badge/VectorDB-Pinecone-blue?style=flat&logo=database&logoColor=white)](https://www.pinecone.io)
[![Pinecone](https://img.shields.io/badge/Pinecone-RAG-6610f2?style=flat&logo=pinecone&logoColor=white)](https://www.pinecone.io)
[![RAG](https://img.shields.io/badge/RAG-Retrieval_Augmented_Generation-6f42c1?style=flat&logo=ai&logoColor=white)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)



---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Scoring System](#scoring-system)
- [LLM Strategy](#llm-strategy)
- [Caching & Resilience](#caching--resilience)
- [Pinecone RAG Layer](#pinecone-rag-layer)
- [Setup & Running Locally](#setup--running-locally)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [File-by-File Breakdown](#file-by-file-breakdown)

---

## Overview

Recruiters spend hours manually screening resumes. This system automates the full pipeline:

1. **Parse** a job description (typed or uploaded) → structured skills + experience range
2. **Parse** all resumes in the `resumes/` folder → structured candidate profiles (with caching + retry + quarantine)
3. **Embed** both JD and candidates using SentenceTransformers
4. **RAG retrieval** via Pinecone (optional) to semantically pre-filter candidates
5. **Score** every candidate with a hybrid formula (skills + projects + experience + tiebreakers)
6. **Rank** and select Top-K candidates
7. **Simulate** candidate interest using Gemini (Top 5 only, to control cost)
8. **Display** results on a dark-themed Streamlit dashboard with skill chip breakdown and score cards

---

## Architecture

```
flowchart TD

A[Job Description — text or file upload]
C[resumes/ folder — PDF, DOCX, TXT]

A --> B[jd_parser.py · Gemini LLM + regex fallback]
C --> D[resume_parser.py · Gemini LLM + retry + quarantine + cache]

B --> E[Structured JD]
D --> F[Structured Candidates]

E --> G[agent.py · SentenceTransformers embed]
F --> G

G --> H[vector_store.py · Pinecone upsert + RAG query]
H --> I[rag_scores per candidate]

G --> J[scorer.py · Hybrid Scoring Engine v4]
I --> J

J --> K[Ranked candidate list]
K --> L[Interest simulation · Gemini · Top 5 only]
L --> M[Final output dict]

M --> N[main.py · FastAPI · /run-agent]
N --> O[UI.py · Streamlit Dashboard]
```

---

## Project Structure

```
ai-talent-agent/
├── app/
│   ├── main.py           # FastAPI backend — routes, JD text extraction from upload
│   ├── agent.py          # Orchestration — ties all modules together
│   ├── scorer.py         # Hybrid scoring engine v4
│   ├── jd_parser.py      # JD parsing — Gemini LLM + regex fallback
│   ├── resume_parser.py  # Resume parsing — Gemini LLM + retry + quarantine
│   ├── vector_store.py   # Pinecone RAG — upsert + semantic retrieval
│   ├── simulator.py      # (Legacy) standalone interest simulator — not used by agent.py
│   └── utils.py          # (Legacy) standalone helpers — not used by agent.py
├── resumes/              # Drop resume files here (PDF, DOCX, TXT)
├── resumes_failed/       # Auto-created — quarantined resumes that failed parsing
├── UI.py                 # Streamlit recruiter dashboard
├── requirements.txt      # All Python dependencies
├── resumes_cache.json    # Auto-generated — parsed resume cache (do not commit)
└── .env                  # API keys (do not commit)
```

> **Note:** `simulator.py` and `utils.py` are legacy files from v1. The active pipeline uses the interest simulation logic embedded directly in `agent.py` and the caching logic inside `agent.py`/`resume_parser.py`.

---

## Tech Stack

| Component | Technology |
|---|---|
| **Backend API** | FastAPI + Uvicorn |
| **Frontend** | Streamlit (dark theme, custom CSS) |
| **LLM** | Google Gemini (`gemini-2.5-flash` by default) via `google-genai` SDK |
| **Embeddings** | SentenceTransformers `all-MiniLM-L6-v2` (384-dim) |
| **Vector DB / RAG** | Pinecone (optional — graceful degradation if unconfigured) |
| **Resume parsing** | PyMuPDF (`fitz`) for PDF, `python-docx` for DOCX |
| **Language** | Python 3.10+ |

---

## Scoring System

### Formula — Hybrid Semantic Scoring Engine v4

```
match_score = 0.60 × skill_score
            + 0.25 × project_score
            + 0.15 × experience_score
            + context_tiebreaker  (max +3%)
            + rag_tiebreaker      (max +3%)

final_score = match_score   ← interest_score is informational only, NOT in ranking
```

### Component Details

#### 1. Skill Score (60%) — Semantic + Exact
The most heavily weighted component because it directly answers "can this person do the job?"

Matching proceeds in order, stopping at the first hit:
1. **Exact match** — after normalization (lowercase, strip parens, normalize separators)
2. **Substring** — one skill contains the other (e.g. "PyTorch" in "PyTorch 2.0")
3. **Token overlap** — ≥75% of JD skill tokens appear in candidate skill
4. **Semantic cosine** — SentenceTransformers embedding cosine ≥ 0.60 threshold

Within the skill component: **70% required skills**, **30% optional skills**.

Both embedding results and match verdicts are cached in-memory to avoid redundant API calls.

#### 2. Project Score (25%) — Embedding
Compares the concatenated text of a candidate's project descriptions against the JD's required skills + keywords via cosine similarity. Captures real-world demonstrated ability beyond listed skills.

#### 3. Experience Score (15%) — Rule-Based

| Condition | Score |
|---|---|
| Within JD range | 1.0 |
| Under by N years | `max(0.0, 1.0 − N × 0.20)` |
| Over by N years | `max(0.60, 1.0 − N × 0.03)` |
| No experience field | 0.65 (neutral, not penalised) |
| JD has no range specified | 0.85 (assumed acceptable) |

Over-qualification carries only a mild penalty — being senior is not a disqualifier.

#### 4. Tiebreakers (max +3% each)
- **Context tiebreaker**: cosine similarity of full candidate bio+skills vs. the JD embedding
- **RAG tiebreaker**: Pinecone cosine score for the candidate, scaled to max +3%

Both tiebreakers are capped to ensure they cannot flip a decision band — they only break ties within a band.

### Decision Thresholds

| Decision | Score Range |
|---|---|
| **Strong Shortlist** ✦ | ≥ 72% |
| **Shortlist** ● | ≥ 58% |
| **Consider** ◐ | ≥ 42% |
| **Reject** ○ | < 42% |

---

## LLM Strategy

LLMs (Gemini) are called **only where structured reasoning or free-text generation is essential** — not for scoring, which is pure math.

| Task | LLM Used? | Why |
|---|---|---|
| JD parsing | ✅ | Structured extraction from freeform text |
| Resume parsing | ✅ | Extracting skills, projects, exp from varied formats |
| Skill scoring | ❌ | Embedding cosine is cheaper, faster, consistent |
| Project scoring | ❌ | Embedding similarity |
| Experience scoring | ❌ | Rule-based — no ambiguity in numbers |
| Interest simulation | ✅ Top 5 only | Generative — needs creative text output |

**Model**: `gemini-2.5-flash` (configurable via `GEMINI_MODEL` env var)

**Interest score derivation** (in `agent.py`): The LLM generates a 2–3 sentence simulated candidate response. The score is then derived heuristically by counting positive sentiment keywords ("excited", "thrilled", "perfect fit" → 0.85; neutral signals → 0.55; etc.). The interest score is **displayed on the UI card** but is **not included in the ranking formula** because it is synthetic and always skews positive.

---

## Caching & Resilience

### Resume Parsing Cache (`resumes_cache.json`)
- **First run**: every resume is parsed by Gemini → result stored in cache keyed by filename
- **Subsequent runs**: cache hit → zero LLM calls, instant load
- Manually delete `resumes_cache.json` to force re-parsing

### Retry Logic (in `resume_parser.py`)
- Each resume gets up to **3 LLM parse attempts** with 1.5s delay between retries
- On all 3 failures: **regex fallback** kicks in (extracts name, email, skills via patterns)
- The failed file is **quarantined** to `resumes_failed/` and re-attempted at end of run

### JSON Robustness (both parsers)
Both `jd_parser.py` and `resume_parser.py` apply the same cleaning pipeline before `json.loads()`:
- Strip markdown fences (` ```json `)
- Remove JS-style `//` and `/* */` comments
- Remove trailing commas before `}` or `]`

---

## Pinecone RAG Layer

`vector_store.py` implements an optional Pinecone-backed semantic retrieval layer.

**When enabled** (requires `PINECONE_API_KEY`):
1. All parsed candidates are embedded and **upserted** to a Pinecone index (`talent-agent`, 384-dim cosine)
2. Each run is **namespaced** by a hash of the JD text, so multiple job runs don't pollute each other
3. The JD is queried against Pinecone to get a **RAG score** (cosine similarity) per candidate
4. This score is passed to `scorer.py` as the `rag_score` tiebreaker (max +3%)

**When disabled** (no API key or `pinecone` not installed): the system degrades gracefully — all RAG scores are `None`, the rag tiebreaker is skipped, and the pipeline runs normally. The UI shows `⚪ Pinecone not configured`.

**Index config**: `talent-agent` on AWS `us-east-1`, serverless, created automatically on first run.

---

## Setup & Running Locally

### 1. Clone the repo
```bash
git clone https://github.com/jhapiyush44/AI-Talent-Agent.git
cd AI-Talent-Agent
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note on PyTorch**: `sentence-transformers` requires `torch`. If the above is slow or you don't need GPU, install the CPU-only build first:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

### 4. Create `.env`
```env
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional — enables Pinecone RAG
PINECONE_API_KEY=your_pinecone_api_key_here

# Optional — override defaults
GEMINI_MODEL=gemini-2.5-flash
RESUME_DIR=resumes
API_URL=http://127.0.0.1:8000
```

### 5. Add resumes
Drop PDF, DOCX, or TXT resume files into the `resumes/` folder.

### 6. Start the FastAPI backend
```bash
uvicorn app.main:app --reload
```
Backend runs at `http://127.0.0.1:8000`. Verify: `http://127.0.0.1:8000/health`

### 7. Start the Streamlit UI (new terminal)
```bash
streamlit run UI.py
```
UI opens at `http://localhost:8501`

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GOOGLE_API_KEY` | ✅ | — | Gemini API key (also accepted as `GEMINI_API_KEY`) |
| `PINECONE_API_KEY` | ❌ | — | Enables Pinecone RAG; system works without it |
| `GEMINI_MODEL` | ❌ | `gemini-2.5-flash` | Gemini model name |
| `RESUME_DIR` | ❌ | `resumes/` | Path to resumes folder |
| `API_URL` | ❌ | `http://127.0.0.1:8000` | Backend URL used by Streamlit UI |

---

## API Reference

### `GET /health`
Health check. Returns `{"status": "ok", "version": "2.0.0"}`.

### `POST /run-agent`
Run the full pipeline.

**Form fields:**

| Field | Type | Description |
|---|---|---|
| `jd_text` | `str` | Raw job description text (use this OR `jd_file`) |
| `jd_file` | `UploadFile` | JD file (PDF, DOCX, or TXT) — text is extracted server-side |
| `top_k` | `int` | Number of top candidates to return (default: 10; use 100 for "All") |

**Response:**
```json
{
  "top_candidates": [
    {
      "name": "Jane Doe",
      "email": "jane@example.com",
      "experience_years": 2.5,
      "skills": ["Python", "PyTorch", "FastAPI", "MLOps"],
      "match_score": 0.81,
      "interest_score": 0.85,
      "final_score": 0.81,
      "decision": "Strong Shortlist",
      "explanation": "─── Scoring Formula ─...",
      "simulated_response": "This role looks like a great fit for me...",
      "interest_reason": "LLM simulated",
      "rag_boosted": true
    }
  ],
  "total_evaluated": 12,
  "jd_required_skills": ["Python", "PyTorch", "FastAPI"],
  "jd_optional_skills": ["Docker", "AWS"],
  "pinecone_enabled": true
}
```

---

## File-by-File Breakdown

### `app/main.py`
FastAPI app (v2.0.0). Sets up CORS, configures logging, and exposes two routes:
- `GET /health` — uptime check
- `POST /run-agent` — entry point for the full pipeline

Also contains `_extract_text_from_upload()` which handles PDF, DOCX, and TXT JD file uploads server-side using PyMuPDF and python-docx.

### `app/agent.py`
The **orchestration layer**. Runs in 9 clearly commented steps:
1. Parse JD
2. Load + parse resumes (with cache)
3. Embed the JD
4. Pinecone RAG upsert + query
5. Score all candidates
6. Rank by match_score
7. Interest simulation (Top 5 only)
8. Compute final_score + decision label
9. Clean internal fields and return API response dict

Also owns the embedding singleton (`all-MiniLM-L6-v2`), the LLM client shim, and the interest simulation logic including the heuristic sentiment scoring.

### `app/scorer.py`
**Hybrid Scoring Engine v4.** Contains:
- `_skill_matches()` — 4-level matching (exact → substring → token overlap → embedding cosine)
- `skill_score()` — semantic skill scoring with required/optional split
- `experience_score()` — rule-based experience scoring with asymmetric penalties
- `compute_match_score()` — master function combining all components + tiebreakers + human-readable explanation
- `decide()` — maps score to decision label

In-memory caches for embeddings (`_embed_cache`) and match results (`_match_cache`) prevent redundant computation within a single run.

### `app/jd_parser.py`
Parses a freeform job description into a structured dict. Uses Gemini with a strict JSON-only prompt, then applies a robust cleaning pipeline (`_clean_json_string`) before `json.loads()`. Falls back to `_fallback_extract()` (regex-based) on any failure. Skill names are preserved as-is (no `.title()` to avoid mangling `MLOps → Mlops`, `GCP → Gcp`).

### `app/resume_parser.py`
Parses individual resume files (PDF via PyMuPDF, DOCX via python-docx, TXT natively). The `parse_resumes()` function implements:
- **Cache-first loading** — skips LLM for already-parsed files
- **Retry loop** — up to 3 LLM attempts per resume with exponential delay
- **Quarantine** — failed files moved to `resumes_failed/` and re-attempted at end of run
- **Fallback parser** — regex-based extraction as last resort

The LLM prompt explicitly instructs the model to: count only paid work experience (not education years), extract skills from every section including project descriptions and interests, and include both full names and abbreviations.

### `app/vector_store.py`
Wraps Pinecone for semantic candidate search. Implements a `VectorStore` class with:
- `upsert_candidates()` — embeds all candidates and upserts to the `talent-agent` index, keyed by MD5 of source filename
- `query_similar()` — retrieves top-K candidates semantically similar to the JD
- `clear_namespace()` — clean-slate per job run
- Graceful degradation if `pinecone` is not installed or `PINECONE_API_KEY` is absent

The index auto-creates on first run (serverless, AWS us-east-1, cosine metric, 384 dimensions).

### `app/simulator.py`
Legacy v1 standalone interest simulator. Uses the old `google.generativeai` SDK directly and returns a JSON dict with `response`, `interest_score`, and `reason`. **Not used by the current pipeline** — interest simulation is now handled inside `agent.py` using the new `google-genai` SDK.

### `app/utils.py`
Legacy v1 utility module with PDF/DOCX text extraction (using `PyPDF2`) and cache load/save helpers. **Not used by the current pipeline** — `resume_parser.py` uses PyMuPDF directly, and cache management is in `agent.py`.

### `UI.py`
Streamlit dashboard (599 lines, v2). Key features:
- **Dark theme** with custom CSS — JetBrains Mono + Space Grotesk fonts, CSS variable palette
- **Left sidebar**: backend status dot, Top-K selector, JD input mode toggle, live pipeline log panel, reset button
- **Main area**: JD input (text or file upload), Run button, last-run stats panel
- **Candidate cards**: colored left border by decision, rank badge, name/email/experience, skill chips (matched vs other), 4-column score display, progress bar, expandable detail tabs (Score Breakdown + Candidate Response)
- **RAG badge** (`RAG ↑`) shown on cards where Pinecone boosted the score
- Progress bar with 7 pipeline step messages during the API call
- 300s request timeout

---

## Future Improvements

- 🗃️ **Multi-JD support** — run agent against multiple roles simultaneously
- 📊 **Batch export** — download results as CSV / PDF report
- 🧠 **Fine-tuned embeddings** — domain-specific model for tech skill matching
- 🔄 **Feedback loop** — recruiter accept/reject signals to improve scoring weights
- ☁️ **Cloud resume storage** — S3/GCS instead of local `resumes/` folder
- 🔍 **Explainability UI** — highlight which resume text triggered each matched skill

---

## 👨‍💻 Author

**Piyush Jha** — ML Engineer  
[GitHub](https://github.com/jhapiyush44) · [LinkedIn](https://www.linkedin.com/in/piyush-jha-3904a81a6/) · jhapiyush44@gmail.com

---

*Found this useful? Consider leaving a ⭐ — it helps others discover the work!*
