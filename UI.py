import streamlit as st
import requests
import time

st.set_page_config(page_title="AI Talent Agent", layout="wide")

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
                "http://127.0.0.1:8000/run-agent",
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

    # ---------- RESULTS ----------
    for i, c in enumerate(result["top_candidates"], 1):

        decision_color = get_color(c["decision"])
        icon = get_icon(c["decision"])

        # ---------- CLEAN CARD ----------
        st.markdown(f"""
        <div style='
            border:1px solid #ddd;
            padding:16px;
            border-radius:10px;
            margin-bottom:12px;
            background-color:#ffffff;
        '>

            <h3 style='color:#111; margin-bottom:4px;'>
                {i}. {c.get('name', 'Unknown Candidate')}
            </h3>

            <p style='margin:0; color:#555; font-size:14px;'>
                📧 {c.get("email", "Not Provided")}
            </p>

            <p style='margin-top:8px; font-weight:600; color:{decision_color};'>
                {icon} {c["decision"]}
            </p>

        </div>
        """, unsafe_allow_html=True)

        # ---------- SCORES ----------
        col1, col2, col3 = st.columns(3)

        col1.metric("Match Score", c["match_score"])
        col2.metric("Interest Score", c["interest_score"])
        col3.metric("Final Score", c["final_score"])

        st.progress(c["final_score"])

        # ---------- DETAILS ----------
        with st.expander("📊 View Details"):
            st.write("**Explanation:**")
            st.write(c["explanation"])

            st.write("**Simulated Candidate Response:**")
            st.write(c["simulated_response"])

            if "interest_reason" in c:
                st.write("**Interest Reason:**")
                st.write(c["interest_reason"])

        st.markdown("---")