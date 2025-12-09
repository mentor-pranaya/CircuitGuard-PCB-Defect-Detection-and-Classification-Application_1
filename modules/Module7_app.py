import streamlit as st
from PIL import Image
import cv2
import numpy as np
import time
import pandas as pd
import io
from datetime import datetime
from backend import CircuitGuardBackend

# ------------------------- PAGE CONFIG -------------------------
st.set_page_config(
    page_title="CircuitGuard AOI | Module 7",
    page_icon="⚡",
    layout="wide"
)

# ------------------------- DARK THEME CSS -------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
}
h1, h2, h3 {
    color: #58a6ff !important;
    text-shadow: 0 0 12px rgba(88,166,255,0.7);
}
.section-box {
    background: #161b22;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #21262d;
    box-shadow: 0px 0px 12px rgba(88,166,255,0.08);
}
div.stButton > button {
    background: linear-gradient(90deg,#238636,#2ea043);
    color: white;
    border-radius: 10px;
    padding: 12px;
    border: none;
    font-weight: bold;
    font-size: 16px;
}
div.stButton > button:hover {
    background: linear-gradient(90deg,#2ea043,#3fb950);
}
.dataframe th, .dataframe td {
    color: white !important;
    background-color: #161b22 !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------- HEADER -------------------------
st.markdown("<h1>⚡ CircuitGuard AOI System</h1>", unsafe_allow_html=True)
st.markdown("<h3>Milestone 4 – Module 7: Final Delivery & Export</h3>", unsafe_allow_html=True)
st.markdown("---")

# ------------------------- BACKEND -------------------------
if "backend" not in st.session_state:
    with st.spinner("Loading AI Engine..."):
        st.session_state.backend = CircuitGuardBackend()
    st.success("AI Engine Ready ✓")

# ------------------------- SESSION STATE -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------- SIDEBAR -------------------------
with st.sidebar:
    st.markdown("<h2>📌 Module 7 Features</h2>", unsafe_allow_html=True)
    st.write("• Full PCB inspection")
    st.write("• Annotated output generation")
    st.write("• CSV defect report export")
    st.write("• Downloadable visual proof")

    if st.session_state.history:
        df_log = pd.DataFrame(st.session_state.history)
        csv_log = df_log.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📄 Download Session Log",
            data=csv_log,
            file_name=f"CircuitGuard_SessionLog_{datetime.now().strftime('%H%M')}.csv",
            mime="text/csv"
        )

# ------------------------- LAYOUT -------------------------
col1, col2 = st.columns([1, 2])

# ------------------------- LEFT: UPLOAD + RUN -------------------------
with col1:
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.subheader("📂 Upload PCB Image")

    test_file = st.file_uploader("Choose Image", type=["png","jpg","jpeg"])

    if test_file:
        img_pil = Image.open(test_file).convert("RGB")
        st.image(img_pil, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 START INSPECTION"):
            with st.spinner("Analyzing PCB..."):
                start = time.time()
                viz, results, msg = st.session_state.backend.run_pipeline(img_pil)
                end = time.time()

                st.session_state.viz = viz
                st.session_state.results = results
                st.session_state.msg = msg
                st.session_state.time = round(end - start, 2)

                st.session_state.history.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Filename": test_file.name,
                    "Defects Found": len(results),
                    "Status": msg,
                    "Processing Time": st.session_state.time
                })

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------- RIGHT: RESULTS -------------------------
with col2:
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.subheader("🎯 Inspection Results")

    if "viz" in st.session_state:

        if st.session_state.msg != "Success":
            st.error(f"❌ {st.session_state.msg}")
        else:
            viz_rgb = cv2.cvtColor(st.session_state.viz, cv2.COLOR_BGR2RGB)
            st.image(viz_rgb, caption="Annotated Output", use_column_width=True)

            count = len(st.session_state.results)
            m1, m2, m3 = st.columns(3)
            m1.metric("Status", "FAIL" if count > 0 else "PASS")
            m2.metric("Defects", count)
            m3.metric("Time", f"{st.session_state.time}s")

            if count > 0:
                df = pd.DataFrame(st.session_state.results)
                st.table(df)

                # ----------------- DOWNLOAD CSV -----------------
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📄 Download Defect Report (CSV)",
                    data=csv,
                    file_name=f"Defects_{test_file.name.replace('.', '_')}.csv",
                    mime="text/csv"
                )

                # ----------------- DOWNLOAD ANNOTATED IMAGE -----------------
                img_bytes = io.BytesIO()
                Image.fromarray(viz_rgb).save(img_bytes, format="PNG")
                st.download_button(
                    label="🖼️ Download Annotated Image",
                    data=img_bytes.getvalue(),
                    file_name=f"Annotated_{test_file.name}",
                    mime="image/png"
                )
            else:
                st.success("Board is clean ✓")

    st.markdown("</div>", unsafe_allow_html=True)
