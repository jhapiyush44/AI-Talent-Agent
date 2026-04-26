from app.jd_parser import parse_jd
from app.scorer import compute_match
from app.simulator import simulate_interest


def run_agent(jd_text, candidates, top_k=5):
    jd = parse_jd(jd_text)

    if not candidates:
        return {
            "parsed_jd": jd,
            "top_candidates": [],
            "all_candidates": [],
            "error": "No candidates available"
        }

    results = []

    # ---------- STEP 1: SCORING ----------
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        print(f"Processing candidate: {candidate.get('name', 'Unknown')}")

        match_score, explanation = compute_match(candidate, jd)

        results.append({
            "candidate": candidate,
            "match_score": match_score,
            "explanation": explanation
        })

    # ---------- STEP 2: RANK ----------
    ranked = sorted(results, key=lambda x: x["match_score"], reverse=True)

    top_candidates = ranked[:top_k]
    rest_candidates = ranked[top_k:]

    # dynamic threshold
    threshold = max(0.3, top_candidates[-1]["match_score"]) if top_candidates else 0.3

    final_results = []

    # ---------- STEP 3: PROCESS TOP CANDIDATES ----------
    for i, item in enumerate(top_candidates):
        candidate = item["candidate"]
        match_score = item["match_score"]
        explanation = item["explanation"]

        try:
            interest_data = simulate_interest(candidate, jd)
            interest_score = interest_data.get("interest_score", 0.5)
            interest_score = max(0, min(interest_score, 1))

            simulated_response = interest_data.get("response", "")
            interest_reason = interest_data.get("reason", "")

        except:
            interest_score = 0.5
            simulated_response = "No response available"
            interest_reason = "Fallback"

        final_score = (0.75 * match_score) + (0.25 * interest_score)

        decision = get_decision(final_score)

        final_results.append(format_candidate(
            i, candidate, match_score, interest_score, final_score,
            decision, explanation, simulated_response, interest_reason
        ))

    # ---------- STEP 4: PROCESS REST (NO LLM) ----------
    for i, item in enumerate(rest_candidates, start=len(top_candidates)):
        candidate = item["candidate"]
        match_score = item["match_score"]
        explanation = item["explanation"]

        interest_score = 0.3 if match_score < threshold else 0.5

        final_score = (0.75 * match_score) + (0.25 * interest_score)

        decision = get_decision(final_score)

        final_results.append(format_candidate(
            i, candidate, match_score, interest_score, final_score,
            decision, explanation, "Not evaluated", "Skipped LLM"
        ))

    # ---------- FINAL SORT ----------
    final_ranked = sorted(final_results, key=lambda x: x["final_score"], reverse=True)

    return {
        "parsed_jd": jd,
        "top_candidates": final_ranked[:top_k],
        "all_candidates": final_ranked
    }


# ---------- HELPERS ----------
def get_decision(score):
    if score >= 0.75:
        return "Strong Shortlist"
    elif score >= 0.60:
        return "Shortlist"
    elif score >= 0.45:
        return "Consider"
    else:
        return "Reject"


def format_candidate(i, candidate, match_score, interest_score, final_score,
                     decision, explanation, response, reason):

    explanation_clean = f"""
Match Score: {round(match_score, 2)}
Interest Score: {round(interest_score, 2)}
Final Score: {round(final_score, 2)}

{explanation}
""".strip()

    return {
        "id": i,
        "name": candidate.get("name", "Unknown Candidate"),
        "email": candidate.get("email", "Not Provided"),
        "skills": candidate.get("skills", []),
        "experience": candidate.get("experience", 0),

        "match_score": round(match_score, 2),
        "interest_score": round(interest_score, 2),
        "final_score": round(final_score, 2),

        "decision": decision,
        "explanation": explanation_clean,
        "simulated_response": response,
        "interest_reason": reason
    }