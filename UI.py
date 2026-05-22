"""
UI.py — AI Talent Agent · Streamlit Dashboard v2
Dark theme · Left sidebar controls · Right panel live logs · Rich candidate cards
"""

import streamlit as st
import requests
import time
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Talent Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
  }

  /* ── Root palette ── */
  :root {
    --bg-deep:    #0a0c10;
    --bg-panel:   #111318;
    --bg-card:    #161920;
    --bg-hover:   #1c1f28;
    --border:     #252932;
    --border-glow:#2e4a7a;
    --accent:     #3d8ef0;
    --accent-dim: #1e3a5f;
    --green:      #22c55e;
    --yellow:     #eab308;
    --orange:     #f97316;
    --red:        #ef4444;
    --text-1:     #f0f2f5;
    --text-2:     #9ca3af;
    --text-3:     #4b5563;
    --mono:       'JetBrains Mono', monospace;
  }

  /* ── Global background ── */
  .stApp { background-color: var(--bg-deep); }
  .block-container { padding: 1.5rem 1.5rem 3rem; max-width: 1600px; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background-color: var(--bg-panel) !important;
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] .stMarkdown p,
  [data-testid="stSidebar"] label { color: var(--text-2) !important; }

  /* ── Headings ── */
  h1, h2, h3 { color: var(--text-1) !important; }

  /* ── Text area ── */
  textarea {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-1) !important;
    font-family: var(--mono) !important;
    font-size: 13px !important;
    border-radius: 8px !important;
  }
  textarea:focus { border-color: var(--accent) !important; outline: none !important; }

  /* ── Selectbox / radio ── */
  [data-testid="stSelectbox"] > div,
  [data-testid="stRadio"]     > div { color: var(--text-1) !important; }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #2563c7 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
    transition: all .2s ease !important;
    box-shadow: 0 0 12px rgba(61,142,240,0.25) !important;
  }
  .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 0 24px rgba(61,142,240,0.45) !important;
  }

  /* ── Metrics ── */
  [data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
  }
  [data-testid="stMetricValue"] { color: var(--text-1) !important; font-size: 22px !important; }
  [data-testid="stMetricLabel"] { color: var(--text-2) !important; font-size: 12px !important; }

  /* ── Progress ── */
  [data-testid="stProgressBar"] > div { background-color: var(--accent-dim) !important; border-radius: 99px; }
  [data-testid="stProgressBar"] > div > div { background-color: var(--accent) !important; border-radius: 99px; }

  /* ── Expander ── */
  details { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
  summary { color: var(--text-2) !important; }

  /* ── File uploader ── */
  [data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
  }

  /* ── Alert / info ── */
  [data-testid="stAlert"] { border-radius: 8px !important; }

  /* ── Log panel ── */
  .log-panel {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text-2);
    max-height: 480px;
    overflow-y: auto;
    line-height: 1.7;
  }
  .log-entry { margin: 2px 0; display: flex; gap: 8px; }
  .log-ts    { color: var(--text-3); min-width: 56px; }
  .log-icon  { min-width: 18px; }
  .log-msg   { color: var(--text-1); }
  .log-ok    { color: var(--green); }
  .log-warn  { color: var(--yellow); }
  .log-err   { color: var(--red); }
  .log-info  { color: var(--accent); }

  /* ── Candidate card ── */
  .c-card {
    background: var(--bg-card);
    border-radius: 14px;
    padding: 22px 24px 18px;
    margin-bottom: 18px;
    border-left: 4px solid var(--border);
    transition: border-color .2s;
  }
  .c-card:hover { border-left-color: var(--accent); }
  .c-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 8px;
  }
  .c-rank {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-3);
    letter-spacing: .1em;
  }
  .c-name {
    font-size: 20px;
    font-weight: 700;
    color: var(--text-1);
    margin: 2px 0;
  }
  .c-email {
    font-size: 13px;
    color: var(--text-2);
  }
  .badge {
    padding: 5px 14px;
    border-radius: 99px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .04em;
    white-space: nowrap;
  }
  .badge-strong { background: #14532d; color: #4ade80; border: 1px solid #166534; }
  .badge-shortlist { background: #713f12; color: #fbbf24; border: 1px solid #92400e; }
  .badge-consider { background: #7c2d12; color: #fb923c; border: 1px solid #9a3412; }
  .badge-reject { background: #450a0a; color: #f87171; border: 1px solid #7f1d1d; }
  .skill-chip {
    display: inline-block;
    background: var(--accent-dim);
    color: var(--accent);
    border: 1px solid var(--border-glow);
    border-radius: 6px;
    font-size: 11px;
    font-family: var(--mono);
    padding: 2px 8px;
    margin: 2px 3px 2px 0;
  }
  .skill-chip-miss {
    background: #2d1515;
    color: #f87171;
    border-color: #7f1d1d;
  }
  .divider { border: none; border-top: 1px solid var(--border); margin: 14px 0; }
  .score-label { font-size: 11px; color: var(--text-3); font-family: var(--mono); letter-spacing:.06em; text-transform:uppercase; }
  .score-val   { font-size: 26px; font-weight:700; color: var(--text-1); }
  .rag-badge   { display:inline-block; background:#0d2a45; color:#60a5fa; border:1px solid #1e3a5f; border-radius:6px; font-size:10px; font-family:var(--mono); padding:1px 7px; margin-left:6px; vertical-align:middle; }

  /* ── Section header ── */
  .section-title {
    font-size: 11px;
    font-family: var(--mono);
    letter-spacing: .15em;
    color: var(--text-3);
    text-transform: uppercase;
    margin: 0 0 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: var(--bg-deep); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "logs"   not in st.session_state: st.session_state.logs   = []
if "result" not in st.session_state: st.session_state.result = None
if "running" not in st.session_state: st.session_state.running = False


def add_log(msg: str, kind: str = "info"):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append({"ts": ts, "msg": msg, "kind": kind})


def render_logs(container):
    icon_map = {"ok": "✓", "warn": "⚠", "err": "✕", "info": "›", "step": "◆"}
    rows = ""
    for e in st.session_state.logs:
        k = e["kind"]
        icon = icon_map.get(k, "·")
        css  = f"log-{k}" if k in ("ok","warn","err","info") else "log-info"
        rows += (
            f'<div class="log-entry">'
            f'<span class="log-ts">{e["ts"]}</span>'
            f'<span class="log-icon {css}">{icon}</span>'
            f'<span class="log-msg">{e["msg"]}</span>'
            f'</div>'
        )
    container.markdown(
        f'<div class="log-panel">{rows or "<span style=\'color:#4b5563\'>No logs yet...</span>"}</div>',
        unsafe_allow_html=True,
    )


# ── Backend health ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def check_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except:
        return False


# ── LEFT SIDEBAR ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:12px 0 20px'>
      <p style='font-size:11px;letter-spacing:.2em;color:#4b5563;text-transform:uppercase;margin:0'>AI Talent Agent</p>
      <h2 style='font-size:22px;font-weight:700;color:#f0f2f5;margin:4px 0 0'>Control Panel</h2>
    </div>
    """, unsafe_allow_html=True)

    # Backend status
    healthy = check_health()
    status_color = "#22c55e" if healthy else "#ef4444"
    status_text  = "Backend Online" if healthy else "Backend Offline"
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;padding:8px 12px;background:#111318;'
        f'border:1px solid #252932;border-radius:8px;margin-bottom:16px">'
        f'<span style="width:8px;height:8px;border-radius:50%;background:{status_color};'
        f'box-shadow:0 0 6px {status_color};display:inline-block"></span>'
        f'<span style="font-size:12px;color:{status_color};font-family:\'JetBrains Mono\',monospace">{status_text}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    if not healthy:
        st.error("Start backend: `uvicorn app.main:app --reload`")
        st.stop()

    st.markdown('<p class="section-title" style="color:#4b5563;font-size:10px;letter-spacing:.15em;text-transform:uppercase">Configuration</p>', unsafe_allow_html=True)

    top_k_option = st.selectbox("Top Candidates", ["Top 5", "Top 10", "All"], index=1)
    top_k = 5 if top_k_option == "Top 5" else 10 if top_k_option == "Top 10" else 100

    jd_mode = st.radio("JD Input Mode", ["✍️  Type Text", "📎  Upload File"], label_visibility="visible")

    st.markdown("---")
    st.markdown('<p class="section-title" style="color:#4b5563;font-size:10px;letter-spacing:.15em;text-transform:uppercase;margin-bottom:8px">Live Pipeline Logs</p>', unsafe_allow_html=True)
    log_container = st.empty()
    render_logs(log_container)

    st.markdown("---")
    if st.button("🗑  Clear & Reset", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ── MAIN AREA ─────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div style='text-align:center;padding:32px 0 24px'>
  <p style='font-family:"JetBrains Mono",monospace;font-size:11px;letter-spacing:.25em;
     color:#3d8ef0;text-transform:uppercase;margin:0'>Powered by Gemini + Pinecone RAG</p>
  <h1 style='font-size:40px;font-weight:700;color:#f0f2f5;margin:6px 0 4px;letter-spacing:-.02em'>
    🎯 AI Talent Scouting Dashboard
  </h1>
  <p style='color:#6b7280;font-size:15px;margin:0'>
    Intelligent candidate screening · Hybrid scoring · Semantic retrieval
  </p>
</div>
""", unsafe_allow_html=True)

# ── JD Input ──────────────────────────────────────────────────────────────────
col_jd, col_run = st.columns([3, 1], gap="large")

with col_jd:
    st.markdown('<p class="section-title">Job Description</p>', unsafe_allow_html=True)

    jd_text = ""
    jd_file = None

    if "Type" in jd_mode:
        jd_text = st.text_area(
            "Paste the full job description here",
            height=220,
            placeholder="e.g. We are looking for an AI/ML Engineer with 2+ years experience in Python, "
                        "TensorFlow, MLOps... Required: Python, PyTorch, FastAPI...",
            label_visibility="collapsed",
        )
    else:
        jd_file = st.file_uploader(
            "Upload JD (PDF / DOCX / TXT)",
            type=["pdf", "docx", "txt"],
            label_visibility="collapsed",
        )
        if jd_file:
            st.success(f"✅  Loaded: **{jd_file.name}**")

with col_run:
    st.markdown('<p class="section-title">Actions</p>', unsafe_allow_html=True)
    st.write("")  # spacing
    run_clicked = st.button("▶  Run Agent", use_container_width=True)

    if st.session_state.result:
        res = st.session_state.result
        total = res.get("total_evaluated", 0)
        shown = len(res.get("top_candidates", []))
        pinecone_on = res.get("pinecone_enabled", False)

        st.markdown(
            f'<div style="background:#111318;border:1px solid #252932;border-radius:10px;'
            f'padding:14px 16px;margin-top:10px">'
            f'<p style="margin:0 0 6px;font-size:11px;color:#4b5563;font-family:\'JetBrains Mono\',monospace;'
            f'letter-spacing:.12em;text-transform:uppercase">Last Run</p>'
            f'<p style="margin:0;font-size:13px;color:#9ca3af">Evaluated: <b style="color:#f0f2f5">{total}</b></p>'
            f'<p style="margin:2px 0 0;font-size:13px;color:#9ca3af">Showing: <b style="color:#f0f2f5">{shown}</b></p>'
            f'<p style="margin:6px 0 0;font-size:11px;color:{"#60a5fa" if pinecone_on else "#6b7280"}">'
            f'{"🔵 Pinecone RAG active" if pinecone_on else "⚪ Pinecone not configured"}</p>'
            f'</div>',
            unsafe_allow_html=True
        )

st.markdown("<hr style='border:none;border-top:1px solid #1c1f28;margin:8px 0 20px'>", unsafe_allow_html=True)


# ── RUN LOGIC ─────────────────────────────────────────────────────────────────
if run_clicked:
    if "Type" in jd_mode and not jd_text.strip():
        st.error("⚠️  Please enter a job description.")
        st.stop()
    if "Upload" in jd_mode and not jd_file:
        st.error("⚠️  Please upload a JD file.")
        st.stop()

    st.session_state.logs = []
    add_log("Agent initiated", "info")
    render_logs(log_container)

    files = []
    if jd_file:
        files.append(("jd_file", (jd_file.name, jd_file.getvalue())))

    data = {"jd_text": jd_text, "top_k": top_k}

    steps = [
        ("Parsing job description...",        0.3,  "step"),
        ("Loading & parsing resumes...",       1.2,  "step"),
        ("Computing embeddings...",            0.5,  "step"),
        ("Running Pinecone RAG retrieval...",  0.4,  "step"),
        ("Scoring candidates (hybrid)...",     0.6,  "step"),
        ("Simulating candidate interest...",   0.5,  "step"),
        ("Ranking & finalising results...",    0.3,  "step"),
    ]

    progress_bar = st.progress(0, text="Starting pipeline...")

    try:
        for i, (msg, delay, kind) in enumerate(steps):
            add_log(msg, kind)
            render_logs(log_container)
            progress_bar.progress(int((i + 1) / (len(steps) + 1) * 100), text=msg)
            time.sleep(delay)

        response = requests.post(
            f"{API_URL}/run-agent",
            files=files if files else None,
            data=data,
            timeout=300,
        )
        result = response.json()

        if "error" in result and result["error"]:
            add_log(f"Pipeline error: {result['error']}", "err")
            render_logs(log_container)
            st.error(f"❌  {result['error']}")
            st.stop()

        progress_bar.progress(100, text="Complete ✅")
        n = len(result.get("top_candidates", []))
        add_log(f"Pipeline complete — {n} candidate(s) ranked", "ok")
        add_log(f"Required skills parsed: {', '.join(result.get('jd_required_skills',[])[:5])}", "info")
        if result.get("pinecone_enabled"):
            add_log("Pinecone RAG boost applied", "ok")
        render_logs(log_container)

        st.session_state.result = result

    except Exception as e:
        add_log(f"Request failed: {e}", "err")
        render_logs(log_container)
        st.error(f"❌  API Error: {e}")
        st.stop()


# ── RESULTS ───────────────────────────────────────────────────────────────────
if st.session_state.result:
    result     = st.session_state.result
    candidates = result.get("top_candidates", [])

    if not candidates:
        st.warning("No candidates returned.")
    else:
        st.markdown(
            f'<p style="font-size:13px;color:#6b7280;margin:0 0 18px">'
            f'Showing <b style="color:#f0f2f5">{len(candidates)}</b> of '
            f'<b style="color:#f0f2f5">{result.get("total_evaluated",len(candidates))}</b> evaluated</p>',
            unsafe_allow_html=True
        )

        for i, c in enumerate(candidates, 1):
            decision = c.get("decision", "Consider")

            # Badge style
            badge_class = {
                "Strong Shortlist": "badge-strong",
                "Shortlist":        "badge-shortlist",
                "Consider":         "badge-consider",
                "Reject":           "badge-reject",
            }.get(decision, "badge-consider")

            badge_icon = {
                "Strong Shortlist": "✦",
                "Shortlist":        "●",
                "Consider":         "◐",
                "Reject":           "○",
            }.get(decision, "◐")

            # Border colour
            border_col = {
                "Strong Shortlist": "#22c55e",
                "Shortlist":        "#eab308",
                "Consider":         "#f97316",
                "Reject":           "#ef4444",
            }.get(decision, "#3d8ef0")

            # Skills: split into matched / other
            req_skills     = result.get("jd_required_skills", [])
            cand_skills    = c.get("skills", [])
            matched_skills = [s for s in cand_skills if any(
                s.lower() in r.lower() or r.lower() in s.lower() for r in req_skills
            )]
            other_skills   = [s for s in cand_skills if s not in matched_skills]

            skill_chips = "".join(
                f'<span class="skill-chip">{s}</span>' for s in matched_skills[:10]
            )
            other_chips = "".join(
                f'<span class="skill-chip" style="background:#1a1d26;color:#6b7280;border-color:#252932">{s}</span>'
                for s in other_skills[:8]
            )

            rag_tag = '<span class="rag-badge">RAG ↑</span>' if c.get("rag_boosted") else ""
            exp_str = f'{c["experience_years"]} yrs' if c.get("experience_years") is not None else "N/A"

            st.markdown(f"""
            <div class="c-card" style="border-left-color:{border_col}">
              <div class="c-header">
                <div>
                  <span class="c-rank">#{i:02d} CANDIDATE</span>{rag_tag}
                  <div class="c-name">{c.get("name","Unknown")}</div>
                  <div class="c-email">
                    {"📧 " + c["email"] if c.get("email") else "📧 Email not provided"}
                    &nbsp;·&nbsp; 💼 {exp_str} experience
                  </div>
                </div>
                <span class="badge {badge_class}">{badge_icon} {decision}</span>
              </div>
              <hr class="divider">
              <div style="margin:10px 0 4px">
                <span class="score-label">Matched Skills &nbsp;</span>
                {skill_chips if skill_chips else '<span style="color:#4b5563;font-size:12px">No direct matches</span>'}
              </div>
              <div style="margin-top:4px">
                <span class="score-label">Other Skills &nbsp;</span>
                {other_chips if other_chips else ""}
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Score row
            s_col1, s_col2, s_col3, s_col4 = st.columns(4)
            with s_col1:
                st.markdown(f'<div style="background:#111318;border:1px solid #1c1f28;border-radius:10px;padding:14px;text-align:center">'
                            f'<div class="score-label">Match Score</div>'
                            f'<div class="score-val">{c["match_score"]:.0%}</div>'
                            f'</div>', unsafe_allow_html=True)
            with s_col2:
                st.markdown(f'<div style="background:#111318;border:1px solid #1c1f28;border-radius:10px;padding:14px;text-align:center">'
                            f'<div class="score-label">Interest Score</div>'
                            f'<div class="score-val">{c["interest_score"]:.0%}</div>'
                            f'</div>', unsafe_allow_html=True)
            with s_col3:
                st.markdown(f'<div style="background:#111318;border:1px solid #1c1f28;border-radius:10px;padding:14px;text-align:center">'
                            f'<div class="score-label">Final Score</div>'
                            f'<div class="score-val" style="color:{border_col}">{c["final_score"]:.0%}</div>'
                            f'</div>', unsafe_allow_html=True)
            with s_col4:
                st.markdown(f'<div style="background:#111318;border:1px solid #1c1f28;border-radius:10px;padding:14px;text-align:center">'
                            f'<div class="score-label">Experience</div>'
                            f'<div class="score-val">{exp_str}</div>'
                            f'</div>', unsafe_allow_html=True)

            # Progress bar
            st.progress(c["final_score"])

            # Expander
            with st.expander("📊  Detailed Analysis", expanded=False):
                tab1, tab2 = st.tabs(["Score Breakdown", "Candidate Response"])

                with tab1:
                    st.markdown(
                        f'<pre style="background:#0d0f14;border:1px solid #1c1f28;border-radius:8px;'
                        f'padding:14px;font-family:\'JetBrains Mono\',monospace;font-size:12px;'
                        f'color:#9ca3af;white-space:pre-wrap">{c.get("explanation","")}</pre>',
                        unsafe_allow_html=True
                    )

                with tab2:
                    st.markdown("**💬 Simulated candidate response:**")
                    st.info(c.get("simulated_response") or "Not available")
                    if c.get("interest_reason") not in ("Fallback", "Default fallback", "Skipped LLM", "Pending", None):
                        st.caption(f"Analysis method: {c['interest_reason']}")

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:32px 0 16px;border-top:1px solid #1c1f28;margin-top:32px'>
  <p style='color:#374151;font-size:12px;font-family:"JetBrains Mono",monospace;margin:0'>
    AI TALENT AGENT v2.0 · Gemini + SentenceTransformers + Pinecone RAG
  </p>
</div>
""", unsafe_allow_html=True)