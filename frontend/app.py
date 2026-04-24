import streamlit as st
import requests

API_URL = "http://localhost:5000/analyse"  # adjust port if needed

# -------- PAGE CONFIG --------
st.set_page_config(
    page_title="Plagiarism Detector",
    layout="wide"
)

# -------- TITLE --------
st.title("Plagiarism Detector")
st.write("Upload a PDF to check for plagiarism.")

# -------- FILE UPLOAD --------
uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

# -------- MAIN FLOW --------
if uploaded_pdf:
    with st.spinner("Processing PDF..."):
        response = requests.post(
            API_URL,
            files={"pdf": (uploaded_pdf.name, uploaded_pdf, "application/pdf")}
        )
        print(response.status_code)
        print(response.text)
        result = response.json()
        #result = response.json()
        risk_score = result["risk_score"]
        flags = result["flags"]
        tier_counts = result["tier_counts"]
        breakdown = result["layer_breakdown"]

        # -------- UI DISPLAY --------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(" Plagiarism Score")
            st.metric("Score", f"{risk_score}/100")

        with col2:
            st.subheader(" Flags Summary")
            st.write(
                f" HIGH: {tier_counts['HIGH']} | MEDIUM: {tier_counts['MEDIUM']} | LOW: {tier_counts['LOW']}"
            )
            st.caption(
                f"L1 Lexical: {breakdown['layer1_lexical']} | "
                f"L2 Semantic: {breakdown['layer2_semantic']} | "
                f"L3 Intrinsic: {breakdown['layer3_intrinsic']}"
            )

        # -------- OPTIONAL PREVIEW --------
        st.subheader(" Flagged Sections")

        if flags:
            for f in flags:
                tier_color = {
                    "HIGH": " ",
                    "MEDIUM": " ",
                    "LOW": " "
                }.get(f["tier"], " ")

                with st.expander(
                    f"{tier_color} {f['tier']} — {f['type']} (score: {f['score']:.2f})"
                ):
                    st.write("**Flagged chunk:**", f.get("chunk", "")[:300])

                    if f.get("matched"):
                        st.write("**Matched source:**", f.get("matched", "")[:300])

                    if f.get("section"):
                        st.write(
                            "**Section:**", f["section"],
                            "| **Feature:**", f["feature"],
                            "| **Z-score:**", f["z_score"]
                        )

                    if f.get("explanation"):
                        st.info(f["explanation"])

        else:
            st.success(" No significant plagiarism signals detected.")