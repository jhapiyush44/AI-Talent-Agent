
<div align="center">

# 🎯 AI Talent Agent

> **Intelligent candidate screening powered by Gemini LLM · SentenceTransformers · Pinecone RAG**

*Powered by Gemini 2.5 · Pinecone RAG · Semantic Embeddings · FastAPI · Streamlit*

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

*Drop in a Job Description. Drop in a folder of resumes. Get a ranked, scored, explained shortlist in under 60 seconds.*

</div>


---

## Table of Contents

- [What It Does](#what-it-does)
- [Architecture](#architecture)
- [Scoring Formula](#scoring-formula)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [How the Pipeline Works](#how-the-pipeline-works)
- [Resume & JD Parsing](#resume--jd-parsing)
- [Pinecone RAG — What It Does & Why](#pinecone-rag--what-it-does--why)
- [Failure Handling](#failure-handling)
- [UI Overview](#ui-overview)
- [Known Limitations & Roadmap](#known-limitations--roadmap)

---

## What It Does

AI Talent Agent automates the most tedious part of recruitment — reading through piles of resumes and ranking candidates against a job description. It:

- **Parses any JD** — paste text or upload a PDF/DOCX/TXT file
- **Parses all resumes** in a folder — PDF, DOCX, and TXT supported
- **Semantically matches skills** — no keyword lists needed; `"LLM-powered Systems"` matches `"Large Language Models"` automatically via embedding cosine similarity
- **Scores each candidate** using a justified hybrid formula (skills 60%, projects 25%, experience 15%)
- **Ranks and explains** — every score comes with a breakdown showing exactly which required skills matched and which were missing
- **Simulates candidate interest** — LLM-generated realistic response from the candidate's perspective (informational only, not part of the score)
- **Stores profiles in Pinecone** for semantic retrieval at scale — the foundation for handling 500+ resume libraries

---

## Architecture

The system is built in clearly separated layers. Every request flows strictly top-down; the only upward signal is the final ranked result returned to the UI.

```
  USER INPUTS
  ┌────────────────────┐          ┌────────────────────┐
  │  Job Description   │          │   Resumes folder   │
  │ text/PDF/DOCX/TXT  │          │  PDF · DOCX · TXT  │
  └────────┬───────────┘          └──────────┬─────────┘
           │                                 │
           └──────────────┬──────────────────┘
                          ▼
  ┌──────────────────────────────────────────────────────┐
  │              Streamlit Dashboard  (UI.py)            │
  │   Dark theme · JD input · Live log panel · Cards     │
  └──────────────────────┬───────────────────────────────┘
                         │  HTTP POST /run-agent
                         ▼
  ┌──────────────────────────────────────────────────────┐
  │              FastAPI Backend  (app/main.py)          │
  │     /health · /run-agent · JD file extraction        │
  └──────────────────────┬───────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────┐
  │           Agent Orchestrator  (app/agent.py)         │
  │  parse → embed → upsert → retrieve → score → rank   │
  └───┬───────────────┬──────────────────┬───────────────┘
      │               │                  │
      ▼               ▼                  ▼
  ┌────────┐   ┌────────────┐   ┌──────────────────────┐
  │   JD   │   │  Resume    │   │       Scorer         │
  │ Parser │   │  Parser    │   │                      │
  │        │   │            │   │  Semantic skill      │
  │ Gemini │   │ Gemini LLM │   │  matching via embed  │
  │ LLM +  │   │ 3× retry   │   │  cosine ≥ 0.60       │
  │ regex  │   │ quarantine │   │                      │
  │fallback│   │ JSON cache │   │  Project · Exp score │
  └───┬────┘   └─────┬──────┘   └──────────┬───────────┘
      │               │                     │
      └───────┬───────┘                     │
              ▼                             ▼
  ┌───────────────────────┐   ┌─────────────────────────┐
  │  Google Gemini 2.5    │   │   all-MiniLM-L6-v2      │
  │  Flash                │   │   SentenceTransformers  │
  │  google-genai SDK     │   │   384-dim  ·  local CPU │
  │                       │   │   cached after 1st load │
  │  · JD parsing         │   │                         │
  │  · Resume parsing     │   │   · Skill pair embed    │
  │  · Interest sim       │   │   · Project score       │
  └───────────────────────┘   │   · Context tiebreaker  │
                               └────────────┬────────────┘
                                            │ embeds
                               ┌────────────▼────────────┐
                               │   Vector Store          │
                               │   (vector_store.py)     │
                               │                         │
                               │   Pinecone Serverless   │
                               │   · Upsert profiles     │
                               │   · Keyed by filename   │
                               │   · Namespaced per JD   │
                               │   · Query → RAG scores  │
                               └─────────────────────────┘

  ──────────────────── SCORING PIPELINE ─────────────────────

  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────┐
  │  Skill score     │  │  Project score   │  │  Experience score  │
  │  0.60 weight     │  │  0.25 weight     │  │  0.15 weight       │
  │                  │  │                  │  │                    │
  │  Per JD skill:   │  │  Embed project   │  │  Work yrs only     │
  │  embed + cosine  │  │  descriptions vs │  │  (not grad year)   │
  │  ≥ 0.60 = match  │  │  JD keywords     │  │  null → 0.65       │
  └────────┬─────────┘  └────────┬─────────┘  └──────────┬─────────┘
           └────────────────────┬┘                        │
                                └────────────────┬────────┘
                                                 ▼
                  match = 0.60×skill + 0.25×project + 0.15×experience
                  final = match + context boost (≤3%) + RAG boost (≤3%)
                  interest simulation → shown on card, NOT in formula

  ──────────────────── DECISIONS ─────────────────────────────

  ┌──────────────────┬──────────────┬────────────┬──────────┐
  │ Strong Shortlist │  Shortlist   │  Consider  │  Reject  │
  │     ≥ 72%        │   ≥ 58%      │  ≥ 42%     │  < 42%   │
  └──────────────────┴──────────────┴────────────┴──────────┘
                                 │
                                 ▼
  ┌──────────────────────────────────────────────────────────┐
  │          Ranked candidate cards  →  Streamlit UI         │
  │  Breakdown · matched skills · simulated interest         │
  └──────────────────────────────────────────────────────────┘
```

### Failure handling in the pipeline

Each resume goes through up to three LLM parse attempts with a 1.5s delay between each. On any success the result is cached and scoring proceeds normally. If all three fail, the resume is copied to `resumes_failed/`, a regex fallback result is used for that run, and a final re-attempt happens after the entire primary pass completes. If that recovery succeeds the LLM result replaces the fallback; if it fails the fallback stays in the output.

### Data flow table

| Stage | Input | Output | Module |
|---|---|---|---|
| JD parsing | Raw JD text | Required skills, optional skills, exp range | `jd_parser.py` + Gemini |
| Resume parsing | PDF/DOCX/TXT | Name, skills, projects, experience, summary | `resume_parser.py` + Gemini |
| Embedding | Text strings | 384-dim float vectors | `all-MiniLM-L6-v2` (local) |
| RAG upsert | Candidate profile text | Pinecone index entry, keyed by filename | `vector_store.py` |
| RAG query | JD embedding | Top-K candidate IDs + cosine scores | `vector_store.py` |
| Skill matching | JD skill ↔ resume skill pairs | Match / no-match, per pair | `scorer.py` |
| Scoring | Component scores | `match_score` 0–1 | `scorer.py` |
| Interest sim | Top-5 candidate + JD excerpt | Simulated candidate response text | `agent.py` → Gemini |
| Ranking | All scored candidates | Sorted list + decision badges | `agent.py` |

---

## Scoring Formula

The final score is computed purely from measurable signals. Interest simulation is **shown on the card but excluded from the score** — it's LLM-generated output that always skews positive and would artificially inflate rankings.

```
match_score = 0.60 × skill_score
            + 0.25 × project_score
            + 0.15 × experience_score
            + context_tiebreaker  (max +3%)
            + RAG_tiebreaker      (max +3%)

final_score = match_score
```

### Component breakdown

| Component | Weight | What it measures |
|---|---|---|
| **Skill Score** | 60% | Semantic match of candidate skills against JD required (70%) + optional (30%) skills |
| **Project Score** | 25% | Embedding cosine similarity between project descriptions and JD keywords + required skills |
| **Experience Score** | 15% | Years of work experience vs JD range (None → 0.65 neutral, not penalised) |
| **Context Tiebreaker** | ≤3% | Full profile embedding vs JD embedding — breaks ties within a band |
| **RAG Tiebreaker** | ≤3% | Pinecone cosine score — semantic profile retrieval boost |

### Decision thresholds

| Decision | Threshold | What it means |
|---|---|---|
| **Strong Shortlist** | ≥ 72% | ≥70% required skills matched + strong project evidence |
| **Shortlist** | ≥ 58% | Majority of required skills, reviewable projects |
| **Consider** | ≥ 42% | Partial match — worth manual review |
| **Reject** | < 42% | Insufficient demonstrated skill coverage |

### Skill matching — semantic, not keyword

Every JD skill is embedded using `all-MiniLM-L6-v2` and compared against every candidate skill via cosine similarity. A match is declared when similarity ≥ 0.60. This means:

- `"LLM-powered Systems"` ↔ `"Large Language Models"` → match (~0.91 sim)  
- `"Google Cloud (Learning)"` ↔ `"GCP"` → match (~0.82 sim)
- `"Gemini API"` ↔ `"Gemini"` → match (~0.95 sim)

No synonym tables. No manual maintenance. Works across any domain.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash (`google-genai` SDK) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, CPU) |
| Vector DB | Pinecone Serverless (free tier, `us-east-1`) |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit (custom dark theme, CSS variables) |
| PDF parsing | PyMuPDF (`fitz`) |
| DOCX parsing | `python-docx` |
| Resume/JD extraction | Gemini 2.5 Flash with structured JSON prompts + regex fallback |

---

## Project Structure

```
ai-talent-agent/
│
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app — /health, /run-agent endpoints
│   ├── agent.py          # Orchestration: ties all modules together
│   ├── jd_parser.py      # JD → structured skills/experience (LLM + regex fallback)
│   ├── resume_parser.py  # Resume → structured profile (LLM + retry + quarantine)
│   ├── scorer.py         # Hybrid semantic scoring engine
│   └── vector_store.py   # Pinecone RAG — upsert, query, namespace management
│
├── UI.py                 # Streamlit dashboard
│
├── resumes/              # Drop candidate resumes here (PDF/DOCX/TXT)
├── resumes_failed/       # Auto-populated: resumes that failed all parse attempts
│
├── resumes_cache.json    # Parse cache — delete to force re-parse
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- A [Google AI Studio](https://aistudio.google.com) API key (free)
- A [Pinecone](https://pinecone.io) account and API key (free tier sufficient)

### 1. Clone the repo

```bash
git clone https://github.com/jhapiyush44/AI-Talent-Agent.git
cd AI-Talent-Agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If you see a CUDA warning about an old NVIDIA driver, it's safe to ignore — the embedding model runs on CPU by default and works perfectly.

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

That's all that's required. Everything else has safe defaults.

### 5. Add resumes

Drop all candidate resumes (PDF, DOCX, or TXT) into the `resumes/` folder.

---

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `GOOGLE_API_KEY` | ✅ Yes | — | Google AI Studio API key |
| `PINECONE_API_KEY` | ⚪ Optional | — | Enables RAG retrieval. App works without it |
| `GEMINI_MODEL` | ⚪ Optional | `gemini-2.5-flash` | Swap model without touching code |
| `RESUME_DIR` | ⚪ Optional | `./resumes` | Path to resume folder |
| `API_URL` | ⚪ Optional | `http://127.0.0.1:8000` | Backend URL for the UI |

**Available Gemini models (May 2026):**

| Model | Free RPM | Free RPD | Notes |
|---|---|---|---|
| `gemini-2.5-flash` | 10 | 250 | Recommended — best balance |
| `gemini-2.5-flash-lite` | 15 | 1000 | Highest throughput |
| `gemini-2.5-pro` | 5 | 100 | Most capable |

---

## Running the App

You need two terminal windows.

**Terminal 1 — Start the backend:**

```bash
uvicorn app.main:app --reload
```

**Terminal 2 — Start the UI:**

```bash
streamlit run UI.py
```

Open your browser at `http://localhost:8501`.

> **First run:** The embedding model (`all-MiniLM-L6-v2`) will be downloaded from HuggingFace (~90MB) and cached locally. Subsequent runs load from cache instantly.

> **Cache:** Parsed resumes are cached in `resumes_cache.json`. If you update resumes or want to force a re-parse with improved prompts, delete this file.

---

## How the Pipeline Works

When you click **▶ Run Agent**, the following happens in order:

```
1.  Parse JD          → Gemini extracts required skills, optional skills,
                        experience range, keywords. Regex fallback if JSON fails.

2.  Load Resumes      → Check cache first. Parse uncached resumes with Gemini.
                        On failure: retry up to 3× with 1.5s delay.
                        If all retries fail: quarantine to resumes_failed/,
                        use regex fallback, re-attempt quarantined files at end.

3.  Embed JD          → all-MiniLM-L6-v2 encodes JD summary (384-dim vector)

4.  Pinecone RAG      → Upsert all candidate profile embeddings (keyed by filename).
                        Query index for top-K semantically similar candidates.
                        Returns cosine scores used as tiebreakers.

5.  Score Candidates  → For each candidate × each JD skill:
                          - Fast path: exact / substring / token overlap check
                          - Semantic path: embed both, cosine ≥ 0.60 = match
                        Compute skill / project / experience scores.
                        Apply context + RAG tiebreakers (capped at +3% each).

6.  Rank              → Sort by match_score descending.

7.  Interest Sim      → Gemini generates a realistic candidate response
                        for top-5 only (informational, not scored).

8.  Return results    → JSON response → Streamlit renders candidate cards.
```

---

## Resume & JD Parsing

### JD Parser (`app/jd_parser.py`)

Sends the full JD text to Gemini with a strict prompt instructing it to return bare skill names — not qualified phrases:

```
WRONG: "Generative AI Development", "AI Model Optimization"
RIGHT: "Generative AI", "Model Optimization"
```

The response is cleaned before `json.loads()`:
- Strips markdown fences (` ```json `)
- Removes JS-style `//` and `/* */` comments (common LLM mistakes)
- Removes trailing commas before `}` or `]`

If JSON parsing still fails → regex fallback scans for 30+ known technology keywords.

### Resume Parser (`app/resume_parser.py`)

Key rules injected into the prompt:

- **Experience years:** Only count paid work (jobs/internships). Explicitly instructed to ignore graduation years and education dates. `CGPA: 9.04 | 2022` does not mean 2 years experience.
- **Skills extraction:** Extract from ALL sections including "Bonus", "Learning", "Interests", and project descriptions. If a project says "built with Gemini" → add "Gemini" to skills.
- **Retry logic:** 3 attempts with 1.5s delay between each.
- **Quarantine:** Failed resumes are copied to `resumes_failed/` and re-attempted after all other resumes are processed.

---

## Pinecone RAG — What It Does & Why

### What it does right now

On each run, every candidate's profile (name + summary + skills + projects) is embedded and upserted to a Pinecone serverless index, namespaced by a hash of the JD text. The JD is also embedded and queried against the index. The returned cosine scores are applied as small tiebreaker boosts (max +3%) to candidates who are semantically closest to the JD.

Each candidate is identified by their **source filename** (not name + email), so two resumes from the same person with different filenames correctly produce two distinct Pinecone entries.

### Why it matters at scale

With 8 resumes the RAG step adds minimal value — the tiebreaker barely changes rankings. Its real purpose becomes clear at 500+ resumes:

| Scale | Without Pinecone | With Pinecone |
|---|---|---|
| 8 resumes | Score all 8 directly | Score all 8 directly (same) |
| 500 resumes | 500 × 25 skills × 2 embeds per match = ~25,000 embed calls | Retrieve top 50 via Pinecone → score only 50 → ~2,500 embed calls |
| 5,000 resumes | ~250,000 embed calls, minutes of compute | Top 50 from Pinecone → ~2,500 calls, seconds |

The index persists between runs. Resumes already upserted in previous runs are reused without re-embedding (Pinecone upsert is idempotent — same vector ID = no-op if unchanged).

---

## Failure Handling

| Failure | Behaviour |
|---|---|
| JD JSON parse error | Clean + retry parse; regex fallback if all fail |
| Resume LLM error | Retry 3× with 1.5s delay |
| All retries failed | Quarantine to `resumes_failed/`; regex fallback used for this run |
| Post-quarantine re-attempt | After primary pass, all quarantined files get one more LLM attempt |
| Pinecone not configured | App works normally; RAG tiebreaker = 0 |
| Pinecone auth error | Logged as warning; scoring continues without RAG |
| CUDA not available | Normal — embeddings run on CPU, warning is safe to ignore |

---

## UI Overview

The Streamlit dashboard is fully custom-styled with a dark theme:

- **Left sidebar** — Configuration (top-K, JD input mode), live pipeline log panel with timestamped entries and colour-coded levels (✓ ok, ⚠ warn, ✕ error, › info), backend status indicator, reset button
- **Main area** — JD input (text or file upload), run button, last-run summary
- **Candidate cards** — Ranked with colour-coded border (green → Strong Shortlist, yellow → Shortlist, orange → Consider, red → Reject), skill chips (matched vs unmatched), score metrics, progress bar, RAG boost badge
- **Expandable detail** — Full score breakdown with formula, matched/missing skills, simulated candidate response

---

## Known Limitations & Roadmap

### Current limitations

- Embedding model (`all-MiniLM-L6-v2`) runs on CPU — slow on first load, faster once cached
- Skill matching threshold (0.60) is fixed; some niche technologies may over- or under-match
- Interest simulation is LLM-generated and always sounds positive — treat as illustrative only
- Resume cache (`resumes_cache.json`) must be manually deleted when prompts are updated

### Planned improvements

- [ ] Multi-JD support — score one resume pool against multiple JDs simultaneously
- [ ] Recruiter feedback loop — thumbs up/down on decisions to calibrate thresholds
- [ ] Bulk resume upload via UI (currently requires dropping files in `resumes/` folder)
- [ ] Export shortlist to CSV / PDF report
- [ ] Async scoring — parallel candidate processing for large resume pools
- [ ] GPU embedding support for faster throughput
- [ ] Resume deduplication — detect same person across multiple resume versions

---

## Contributing

Pull requests welcome. For major changes, open an issue first to discuss the approach.

```bash
# Run the backend in dev mode
uvicorn app.main:app --reload --log-level debug

# The UI hot-reloads automatically when you save UI.py
streamlit run UI.py
```

---


<div align="center">
## 👨‍💻 Author

**Piyush Jha** — ML Engineer  
[GitHub](https://github.com/jhapiyush44) · [LinkedIn](https://www.linkedin.com/in/piyush-jha-3904a81a6/) · jhapiyush44@gmail.com

---

*Found this useful? Consider leaving a ⭐ — it helps others discover the work!*

</div>
