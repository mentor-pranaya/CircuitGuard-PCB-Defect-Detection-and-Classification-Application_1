import streamlit as st
from PIL import Image
import cv2
import time
from backend import CircuitGuardBackend

st.set_page_config(page_title="CircuitGuard AI", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.main { background-color: #f5f5f5; }
div.stButton > button { width: 100%; background-color: #007bff; color: white; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ CircuitGuard: PCB Defect Detector")
st.markdown("**Powered by EfficientNet-B4** • Automated Optical Inspection (AOI)")
st.markdown("---")

if "backend" not in st.session_state:
    with st.spinner("🧠 Initializing AI Engine..."):
        try:
            st.session_state.backend = CircuitGuardBackend()
            st.success("System Online: AI Model Loaded")
        except Exception as e:
            st.error(f"Failed to load backend: {e}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📂 Reference Template")
    temp_file = st.file_uploader("Upload Clean PCB", type=['jpg', 'jpeg', 'png'], key="temp")
    if temp_file:
        t_img = Image.open(temp_file).convert('RGB')
        st.image(t_img, use_column_width=True, caption="Reference Standard")

with col2:
    st.subheader("📂 Inspection Target")
    test_file = st.file_uploader("Upload Defective PCB", type=['jpg', 'jpeg', 'png'], key="test")
    if test_file:
        test_img = Image.open(test_file).convert('RGB')
        st.image(test_img, use_column_width=True, caption="Test Unit")

if temp_file and test_file:
    st.markdown("---")
    
    if st.button("🚀 START INSPECTION"):
        bar = st.progress(0)
        status = st.empty()
        
        status.text("Phase 1: Geometric Alignment...")
        bar.progress(25)
        time.sleep(0.2)
        
        status.text("Phase 2: Defect Extraction & Masking...")
        bar.progress(50)
        
        status.text("Phase 3: AI Classification (EfficientNet-B4)...")
        bar.progress(85)

        try:
            annotated_img, data, msg = st.session_state.backend.run_pipeline(t_img, test_img)
        except Exception as e:
            status.error(f"❌ Pipeline Error: {e}")
            st.stop()

        bar.progress(100)
        
        if msg == "Success":
            status.success("✅ Inspection Complete")
            
            res_c1, res_c2 = st.columns([2, 1])
            
            with res_c1:
                st.subheader("🎯 Defect Localization Map")
                viz_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                st.image(viz_rgb, use_column_width=True, caption="AI Annotated Result")
                
            with res_c2:
                st.subheader("📊 Defect Report")
                if data:
                    st.error(f"⚠️ Found {len(data)} Defects")
                    st.table(data)
                else:
                    st.success("✅ Board Passed. No Defects Found.")
        else:
            status.error(f"❌ Inspection Failed: {msg}")
