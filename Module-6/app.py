import streamlit as st
from PIL import Image
import numpy as np

from backend.inference import predict_defects    


st.set_page_config(
    page_title="CircuitGuard | PCB Defect Detection",
    page_icon="🛡️",
    layout="wide"
)

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


st.markdown("<h1 class='title-text'>CircuitGuard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-powered PCB Defect Detection & Classification</p>", unsafe_allow_html=True)

st.write("---")


col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='uploadbox'>", unsafe_allow_html=True)
    template = st.file_uploader("Upload Template PCB Image", type=['png', 'jpg', 'jpeg'])
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='uploadbox'>", unsafe_allow_html=True)
    test = st.file_uploader("Upload Test PCB Image", type=['png', 'jpg', 'jpeg'])
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# PROCESS BUTTON 
if st.button("🔍 Run Defect Detection"):
    
    if template is None or test is None:
        st.error("Please upload both Template and Test images.")
    else:
        with st.spinner("Analyzing PCB images... Please wait "):

            # ---- CALL YOUR BACKEND FUNCTION ----
            annotated_img, logs = predict_defects(template, test)

        st.success("Detection Complete ✔")

        st.markdown("<h3 style='color:#cdd9ff;'>🔧 Results</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#8fb2ff;'>Annotated PCB with detected defects:</p>", unsafe_allow_html=True)

        # DISPLAY THE REAL ANNOTATED OUTPUT 
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.image(annotated_img, caption=" Annotated PCB — Defects Highlighted")
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")

        st.markdown("###  Defect Log")
        st.json(logs)

st.markdown("<p class='footer'>© 2025 CircuitGuard — AI-powered PCB Quality Assurance</p>", unsafe_allow_html=True)
