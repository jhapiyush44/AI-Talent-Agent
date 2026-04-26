import streamlit as st
import requests
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="AI Talent Agent", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- BACKEND HEALTH CHECK ----------
@st.cache_data(ttl=30)
def check_backend_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

if not check_backend_health():
    st.error("⚠️ **Backend server is not running!**")
    st.info(f"""
    Please start the backend server:
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000
    ```
    Expected URL: `{API_URL}`
    """)
    st.stop()

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Settings")

top_k_option = st.sidebar.selectbox(
    "Select Top Candidates",
    ["Top 5", "Top 10", "All"]
)

top_k = 5 if top_k_option == "Top 5" else 10 if top_k_option == "Top 10" else 100

st.sidebar.markdown("---")
st.sidebar.subheader("📡 System Logs")
log_placeholder = st.sidebar.empty()

def update_log(msg):
    log_placeholder.markdown(f"🔹 {msg}")

# ---------- DECISION COLORS ----------
def get_color(decision):
    if decision == "Strong Shortlist":
        return "#2e7d32"  # green
    elif decision == "Shortlist":
        return "#f9a825"  # yellow
    elif decision == "Consider":
        return "#ef6c00"  # orange
    else:
        return "#c62828"  # red

def get_icon(decision):
    if decision == "Strong Shortlist":
        return "✅"
    elif decision == "Shortlist":
        return "🟡"
    elif decision == "Consider":
        return "⚠️"
    else:
        return "❌"

# ---------- HEADER ----------
st.markdown("""
<h1 style='text-align: center;'>🤖 AI Talent Scouting Dashboard</h1>
<p style='text-align: center; color: gray;'>AI-Powered Candidate Ranking & Shortlisting</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------- JD INPUT ----------
jd_mode = st.radio(
    "Choose Job Description Input",
    ["Type as Text", "Upload File"],
    horizontal=True
)

jd_text = ""
jd_file = None

if jd_mode == "Type as Text":
    jd_text = st.text_area("📄 Enter Job Description", height=200)
else:
    jd_file = st.file_uploader("📂 Upload JD (PDF/DOCX)", type=["pdf", "docx"])

st.markdown("---")

# ---------- BUTTONS ----------
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    b1, spacer, b2 = st.columns([1, 0.2, 1])
    run_clicked = b1.button("▶ Run Agent", use_container_width=True)
    reset_clicked = b2.button("🔄 Reset", use_container_width=True)

# ---------- RESET ----------
if reset_clicked:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# ---------- RUN ----------
if run_clicked:

    if jd_mode == "Type as Text" and not jd_text.strip():
        st.error("Please enter Job Description")
        st.stop()

    if jd_mode == "Upload File" and not jd_file:
        st.error("Please upload a JD file")
        st.stop()

    files = []
    if jd_file:
        files.append(("jd_file", (jd_file.name, jd_file.getvalue())))

    data = {"jd_text": jd_text, "top_k": top_k}

    with st.spinner("Running AI Agent..."):

        update_log("Analyzing Job Description...")
        time.sleep(0.5)

        update_log("Matching Candidates...")
        time.sleep(0.5)

        update_log("Ranking Candidates...")
        time.sleep(0.5)

        try:
            response = requests.post(
                f"{API_URL}/run-agent",
                files=files if files else None,
                data=data,
                timeout=180
            )
            result = response.json()
            update_log("Completed ✅")

        except Exception as e:
            st.error(f"API Error: {e}")
            st.stop()

    if "error" in result:
        st.error(result["error"])
        st.stop()

    st.success("✅ Analysis Complete")
    st.markdown("---")

    # ---------- RESULTS ----------
    for i, c in enumerate(result["top_candidates"], 1):

        decision_color = get_color(c["decision"])
        icon = get_icon(c["decision"])

        # ---------- CANDIDATE CARD (Fixed for dark mode) ----------
        st.markdown(f"""
        <div style='
            border: 2px solid {decision_color};
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            background: linear-gradient(135deg, rgba(46,125,50,0.05) 0%, rgba(0,0,0,0) 100%);
        '>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <h2 style='margin: 0; font-size: 24px;'>
                    {i}. {c.get('name', 'Unknown Candidate')}
                </h2>
                <span style='
                    background-color: {decision_color}; 
                    color: white; 
                    padding: 8px 16px; 
                    border-radius: 20px; 
                    font-weight: bold;
                    font-size: 14px;
                '>
                    {icon} {c["decision"]}
                </span>
            </div>
            <p style='margin: 8px 0 0 0; font-size: 14px; opacity: 0.8;'>
                📧 {c.get("email", "Not Provided")}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ---------- SCORES ----------
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("🎯 Match Score", f"{c['match_score']:.2f}", 
                     delta=None, delta_color="off")
        with col2:
            st.metric("💬 Interest Score", f"{c['interest_score']:.2f}", 
                     delta=None, delta_color="off")
        with col3:
            st.metric("⭐ Final Score", f"{c['final_score']:.2f}", 
                     delta=None, delta_color="off")

        # ---------- PROGRESS BAR ----------
        st.progress(c["final_score"], text=f"Overall Score: {c['final_score']:.0%}")

        # ---------- SCORE BREAKDOWN ----------
        with st.expander("📊 View Detailed Analysis", expanded=False):
            
            # Visual breakdown
            st.markdown("**Score Contribution:**")
            match_contribution = c["match_score"] * 0.75
            interest_contribution = c["interest_score"] * 0.25
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Match (75% weight)", f"{match_contribution:.2f}")
            with col_b:
                st.metric("Interest (25% weight)", f"{interest_contribution:.2f}")
            
            st.markdown("---")
            
            # Explanation
            st.markdown("**📈 Match Score Breakdown:**")
            st.text(c["explanation"])
            
            st.markdown("**💬 Simulated Candidate Response:**")
            st.info(c["simulated_response"])
            
            if "interest_reason" in c and c["interest_reason"] not in ["Fallback", "Default fallback", "Skipped LLM"]:
                st.markdown("**🧠 Interest Analysis:**")
                st.write(c["interest_reason"])

        st.markdown("---")