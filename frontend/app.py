import streamlit as st
import fitz  # PyMuPDF

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

# -------- PDF TEXT EXTRACTION --------
def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text

# -------- MAIN FLOW --------
if uploaded_pdf:
    with st.spinner("Processing PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_pdf)

    st.success("PDF uploaded and processed!")

    # -------- PLACEHOLDER RESULTS --------
    # (Backend will replace this later)
    fake_score = 0.0
    highlighted_text = pdf_text  # no highlighting yet

    # -------- UI DISPLAY --------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Plagiarism Score")
        st.metric("Score", f"{fake_score * 100:.2f}%")

    with col2:
        st.subheader("📝 Highlighted Text")
        st.markdown(highlighted_text, unsafe_allow_html=True)

    # -------- OPTIONAL PREVIEW --------
    st.subheader("📃 Extracted Text Preview")
    st.text_area("Text", pdf_text[:2000], height=200)