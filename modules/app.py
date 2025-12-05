import streamlit as st
from PIL import Image
import cv2
import numpy as np
import time
import pandas as pd
from backend import CircuitGuardBackend

st.set_page_config(
    page_title="CircuitGuard AOI",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
}
h1, h2, h3 {
    color: #58a6ff !important;
    text-shadow: 0 0 12px rgba(88,166,255,0.6);
}
.section-box {
    background: #161b22;
    padding: 18px 22px;
    border-radius: 12px;
    border: 1px solid #21262d;
    box-shadow: 0px 0px 12px rgba(88,166,255,0.08);
}
.stFileUploader {
    background: #0d1117 !important;
    padding: 12px;
    border-radius: 10px;
    border: 1px dashed #30363d;
}
div.stButton > button {
    background: linear-gradient(90deg, #238636, #2ea043);
    color: #ffffff;
    border-radius: 10px;
    padding: 10px 18px;
    border: none;
    font-size: 17px;
    font-weight: bold;
    box-shadow: 0px 0px 8px rgba(35,134,54,0.5);
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #2ea043, #3fb950);
    box-shadow: 0px 0px 12px rgba(35,134,54,0.9);
}
.dataframe th, .dataframe td {
    color: #e6edf3 !important;
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>⚡ CircuitGuard AOI System</h1>", unsafe_allow_html=True)
st.markdown("<h3>Smart PCB Defect Detection • EfficientNet-B4 Powered</h3>", unsafe_allow_html=True)

if "backend" not in st.session_state:
    with st.spinner("Booting AI Engine..."):
        st.session_state.backend = CircuitGuardBackend()
    st.success("AI Engine Ready ✓")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.subheader("📂 Upload PCB Image")
    test_file = st.file_uploader("Choose a PCB Image", type=['png', 'jpg', 'jpeg'])

    if test_file:
        img_pil = Image.open(test_file).convert('RGB')
        st.image(img_pil, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 START INSPECTION"):
            with st.spinner("Analyzing Image..."):
                start = time.time()
                viz, results, msg = st.session_state.backend.run_pipeline(img_pil)
                end = time.time()
                st.session_state.viz = viz
                st.session_state.results = results
                st.session_state.msg = msg
                st.session_state.time = round(end - start, 2)

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.subheader("🎯 Inspection Results")

    if "viz" in st.session_state:
        if st.session_state.msg != "Success":
            st.error(f"❌ {st.session_state.msg}")
        else:
            viz_rgb = cv2.cvtColor(st.session_state.viz, cv2.COLOR_BGR2RGB)
            st.image(viz_rgb, caption="Detected Defects", use_column_width=True)

            count = len(st.session_state.results)
            m1, m2, m3 = st.columns(3)
            m1.metric("Board Status", "FAIL" if count > 0 else "PASS")
            m2.metric("Defects Found", count)
            m3.metric("Processing Time", f"{st.session_state.time}s")

            if count > 0:
                df = pd.DataFrame(st.session_state.results)
                st.table(df)
            else:
                st.success("No defects detected ✓")

    st.markdown("</div>", unsafe_allow_html=True)
