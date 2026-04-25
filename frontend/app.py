import streamlit as st
import requests
import time

API_URL = "http://localhost:5000/analyse"

# -------- PAGE CONFIG --------
st.set_page_config(
    page_title="Plagiarism Risk Signal Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------- SIDEBAR --------
with st.sidebar:
    st.header("About")
    st.markdown("""Plagiarism Risk Signal Engine
    
    Analyzes academic manuscripts using three parallel detection layers:
    - **Layer 1**: Lexical (TF-IDF) - verbatim matches
    - **Layer 2**: Semantic (SBERT) - paraphrase detection
    - **Layer 3**: Intrinsic (Stylometry) - style anomalies
    """)
    st.divider()
    st.markdown("**Processing**: 76-paper corpus, parallel detection")

# -------- TITLE & INTRO --------
st.title("Plagiarism Risk Signal Engine")
st.markdown("""
Upload an academic manuscript to detect plagiarism risk signals. The system will 
analyze your paper using three scientifically-validated detection layers and provide 
explainable risk ratings.
""")

# -------- FILE UPLOAD SECTION --------
st.header("Upload Manuscript")
col1, col2 = st.columns([3, 1])

with col1:
    uploaded_pdf = st.file_uploader("Select a PDF file", type=["pdf"])

with col2:
    if uploaded_pdf:
        st.info(f"✓ {uploaded_pdf.name}")
    else:
        st.caption("No file selected")

# -------- ANALYZE BUTTON --------
if uploaded_pdf:
    st.header("Analyze")
    
    analyze_button = st.button(
        "🔍 Analyze for Plagiarism",
        type="primary",
        use_container_width=True
    )
    
    if analyze_button:
        # Show progress
        with st.spinner(""):
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Send to API
            try:
                files = {"pdf": (uploaded_pdf.name, uploaded_pdf, "application/pdf")}
                
                status_placeholder.info("Uploading manuscript...")
                time.sleep(0.5)
                
                status_placeholder.info("Running Stage 1: Ingestion (extracting text & chunking)...")
                time.sleep(0.5)
                
                status_placeholder.info("Running Stage 2: Parallel Detection (3 layers)...")
                time.sleep(0.5)
                
                status_placeholder.info("Running Stage 3: Risk Aggregation & Scoring...")
                time.sleep(0.5)
                
                status_placeholder.info("Running Stage 4: LLM Explanations...")
                
                # Make API call
                response = requests.post(API_URL, files=files, timeout=120)
                
                if response.status_code != 200:
                    st.error(f"❌ API Error: {response.status_code}")
                    st.error(response.text)
                else:
                    result = response.json()
                    status_placeholder.success("✅ Analysis complete!")
                    
                    # Display results
                    st.divider()
                    st.header("Results")
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    risk_score = result["risk_score"]
                    tier_counts = result["tier_counts"]
                    
                    with col1:
                        st.metric("📊 Risk Score", f"{risk_score}/100")
                    
                    with col2:
                        st.metric("🔴 HIGH Flags", tier_counts["HIGH"])
                    
                    with col3:
                        st.metric("🟡 MEDIUM Flags", tier_counts["MEDIUM"])
                    
                    with col4:
                        st.metric("🟢 LOW Flags", tier_counts["LOW"])
                    
                    # Layer breakdown
                    breakdown = result["layer_breakdown"]
                    st.caption(
                        f"Layer 1 (Lexical): {breakdown['layer1_lexical']} | "
                        f"Layer 2 (Semantic): {breakdown['layer2_semantic']} | "
                        f"Layer 3 (Intrinsic): {breakdown['layer3_intrinsic']}"
                    )
                    
                    st.divider()
                    
                    # Risk interpretation
                    if risk_score >= 85:
                        st.error(f"🔴 **HIGH RISK** - Score: {risk_score}/100")
                        st.write("This manuscript shows substantial plagiarism signals that warrant editorial review.")
                    elif risk_score >= 70:
                        st.warning(f"🟡 **MEDIUM RISK** - Score: {risk_score}/100")
                        st.write("This manuscript shows some concerning similarities that should be investigated.")
                    elif risk_score >= 55:
                        st.info(f"🟢 **LOW RISK** - Score: {risk_score}/100")
                        st.write("This manuscript shows minor similarities. Generally acceptable.")
                    else:
                        st.success(f"✅ **CLEAN** - Score: {risk_score}/100")
                        st.write("No significant plagiarism signals detected.")
                    
                    st.divider()
                    
                    # Detailed flags
                    flags = result["flags"]
                    if flags:
                        st.header("Flagged Passages")
                        
                        for i, flag in enumerate(flags, 1):
                            tier = flag['tier']
                            score = flag['score']
                            layer = flag['layer']
                            
                            # Color coding
                            tier_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "NONE": "⚪"}.get(tier, "⚪")
                            
                            with st.expander(
                                f"{tier_emoji} [{tier}] Layer {layer} — Score: {score:.3f} — {flag.get('type', 'Unknown')}"
                            ):
                                # Manuscript chunk
                                st.write("**📝 Submitted Passage:**")
                                st.code(flag.get("chunk", "")[:400], language="text")
                                
                                # Matched source (for L1 & L2)
                                if flag.get("matched") and layer in [1, 2]:
                                    st.write("**🔗 Matched Source:**")
                                    st.code(flag.get("matched", "")[:400], language="text")
                                    
                                    if flag.get("meta"):
                                        meta = flag["meta"]
                                        st.caption(
                                            f"Source: {meta.get('title', 'Unknown')[:60]} | "
                                            f"DOI: {meta.get('doi', 'N/A')}"
                                        )
                                
                                # Style anomaly info (for L3)
                                if layer == 3:
                                    st.write("**🎨 Style Anomaly Details:**")
                                    st.metric("Section", flag.get("section", "Unknown"))
                                    st.metric("Feature", flag.get("feature", "Unknown"))
                                    st.metric("Z-Score", f"{flag.get('z_score', 0):.2f}")
                                
                                # Explanation
                                if flag.get("explanation"):
                                    st.write("**💡 Editorial Note:**")
                                    st.info(flag["explanation"])
                    
                    else:
                        st.success("✅ No significant plagiarism signals detected.")
                    
            except requests.exceptions.Timeout:
                st.error("❌ Analysis timeout (>120s). Try with a shorter document.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API server. Make sure it's running: `python api.py`")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.text(str(e))

else:
    st.info("👆 Please upload a PDF file to begin analysis")