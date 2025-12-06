import streamlit as st
from PIL import Image
import json
import pandas as pd

from backend.inference import predict_defects

# Page UI Config
st.set_page_config(
    page_title="CircuitGuard | PCB Defect Detection",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS
custom_css = """
<style>

body {
    background: linear-gradient(145deg, #0a0f24, #121b36);
    color: white;
}

.sidebar .sidebar-content {
    background: #0e1730;
}

.uploadbox {
    border: 2px dashed #4ea3f1;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    background: rgba(255,255,255,0.04);
    transition: 0.3s;
}

.uploadbox:hover {
    border-color: #00e1ff;
    background: rgba(255,255,255,0.08);
}

.stButton > button {
    background-color: #0066ff;
    color: white;
    padding: 0.6rem 1.4rem;
    border-radius: 10px;
    border: none;
    font-size: 1rem;
    transition: 0.3s;
}

.stButton > button:hover {
    background-color: #0052cc;
}

.result-card {
    background: rgba(255,255,255,0.04);
    padding: 18px;
    border-radius: 15px;
    border: 1px solid #2654a1;
    box-shadow: 0 0 25px rgba(0,128,255,0.3);
}

.title-text {
    font-size: 40px;
    font-weight: 700;
    background: -webkit-linear-gradient(45deg, #00eaff, #007bff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    font-size: 22px;
    color: #b4c6ff;
    margin-bottom: 10px;
}

.footer {
    text-align: center;
    margin-top: 50px;
    color: #8da3d8;
    font-size: 14px;
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Header
st.markdown("<h1 class='title-text'>CircuitGuard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-powered PCB Defect Detection & Classification</p>", unsafe_allow_html=True)
st.write("---")

# Upload Box
st.markdown("<div class='uploadbox'>", unsafe_allow_html=True)
template = st.file_uploader("Upload PCB Image", type=['png', 'jpg', 'jpeg'])
st.markdown("</div>", unsafe_allow_html=True)

# Dropdown for selecting defect type
selected_class = st.selectbox(
    "Select the defect you want to detect:",
    ["Short", "Open", "Mouse Bite", "Spur", "Missing Hole", "Copper Spatter", "Pin Hole"]
)

st.write("")

# RUN BUTTON
if st.button("🔍 Run Defect Detection"):

    if template is None:
        st.error("Please upload a PCB image before running detection.")
    else:
        with st.spinner("Analyzing PCB image... Please wait..."):

            annotated_img, logs = predict_defects(template, selected_class)

        st.success("Detection Complete ✔")

        st.markdown("<h3 style='color:#cdd9ff;'>🔧 Results</h3>", unsafe_allow_html=True)

        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.image(annotated_img, caption="Annotated PCB — Defects Highlighted")
        st.markdown("</div>", unsafe_allow_html=True)

        # Show logs
        st.write("### Detected Defect Log")
        st.json(logs)

        # ------------------------------
        # DOWNLOAD JSON LOG
        # ------------------------------
        log_json = json.dumps(logs, indent=4)

        st.download_button(
            label="📥 Download Log (JSON)",
            data=log_json,
            file_name="pcb_defect_log.json",
            mime="application/json"
        )

        # ------------------------------
        # DOWNLOAD CSV LOG
        # ------------------------------
        if len(logs) > 0:
            log_df = pd.DataFrame(logs)
            csv_data = log_df.to_csv(index=False)

            st.download_button(
                label="📥 Download Log (CSV)",
                data=csv_data,
                file_name="pcb_defect_log.csv",
                mime="text/csv"
            )

# Footer
st.markdown("<p class='footer'>© 2025 CircuitGuard — AI-powered PCB Quality Assurance</p>", unsafe_allow_html=True)
