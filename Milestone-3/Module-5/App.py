"""
CircuitGuard - PCB Defect Detection Web Application
Milestone 3: Module 5 - Frontend with Streamlit
"""
import streamlit as st
import torch
import numpy as np
from PIL import Image
import time

# ==================== Page Configuration ====================
st.set_page_config(
    page_title="CircuitGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit warnings with CSS
st.markdown("""
    <style>
    .stAlert { display: none; }
    </style>
""", unsafe_allow_html=True)

# Import backend functions
try:
    from backend_inference import (
        load_model_and_mapping,
        predict_defect,
        visualize_prediction
    )
except ImportError:
    st.error("Backend inference module not found. Please ensure backend_inference.py is present.")
    st.stop()

# ==================== Professional UI / CSS ====================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    /* Main Background */
    .main {
        background-color: #0a0e1a;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Headers */
    h1 { color: #00d9ff !important; font-family: 'Space Grotesk', sans-serif; font-weight: 700; }
    h2, h3 { color: #e0e7ff !important; }
    p, div, span, label { color: #94a3b8; }
    
    /* Status Banner */
    .status-banner {
        background: linear-gradient(90deg, #1e40af 0%, #7c3aed 100%);
        padding: 1rem; border-radius: 10px; margin-bottom: 2rem;
        border: 1px solid #3b82f6; text-align: center; color: white; font-weight: bold;
    }
    
    /* Result Display */
    .result-display {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 2rem; border-radius: 16px; border: 2px solid #334155;
        margin: 1rem 0; position: relative; overflow: hidden;
    }
    
    .defect-name {
        font-size: 2.5rem; font-weight: 700; color: #fff; margin: 1rem 0;
        text-shadow: 0 0 20px rgba(59, 130, 246, 0.3);
    }
    
    /* Cards */
    .info-card {
        background-color: #1e293b; padding: 1rem; border-radius: 10px;
        border: 1px solid #334155; margin-bottom: 0.5rem;
    }
    .info-card-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
    .info-card-value { font-size: 1rem; color: #e2e8f0; font-weight: 600; }
    
    /* Sidebar & Buttons */
    [data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #1e40af; }
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        color: white; border: none; border-radius: 8px; padding: 0.8rem;
        font-weight: 600; width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== Auto-load model ====================
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

if not st.session_state.model_loaded:
    try:
        with st.spinner("Initializing Enterprise AI System..."):
            model, class_mapping = load_model_and_mapping(
                model_path='best_efficientnet_b4.pth',
                class_mapping_path='class_mapping.json'
            )
            st.session_state.model = model
            st.session_state.class_mapping = class_mapping
            st.session_state.model_loaded = True
    except Exception as e:
        st.sidebar.error(f"System Offline: {str(e)}")

# ==================== Main Application ====================
def main():
    with st.sidebar:
        st.title("CircuitGuard")
        # [REMOVED] Caption "Enterprise Edition v1.1"
        
        if st.session_state.model_loaded:
            st.markdown('<div class="status-banner">SYSTEM ONLINE</div>', unsafe_allow_html=True)
        else:
            if st.button("RETRY CONNECTION"): st.rerun()
            
        st.markdown("### Specifications")
        st.markdown("""
<div class="info-card"><span class="info-card-label">Model</span><div class="info-card-value">EfficientNet-B4</div></div>
<div class="info-card"><span class="info-card-label">Accelerator</span><div class="info-card-value">CUDA / CPU</div></div>
""", unsafe_allow_html=True)

    st.title("PCB Quality Inspector")
    # [REMOVED] Markdown "Automated Defect Detection System"
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload PCB Image", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
    
    if uploaded_file:
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("### Source Feed")
            image = Image.open(uploaded_file).convert('RGB')
            # FIX 1: Use old parameter to prevent crash
            st.image(image, caption="Input Signal", use_column_width=True)
            
            if st.button("START SCAN"):
                with st.spinner("Processing..."):
                    start = time.time()
                    try:
                        pred = predict_defect(st.session_state.model, image, st.session_state.class_mapping)
                        
                        st.session_state.last_res = pred
                        st.session_state.last_img = image
                        st.session_state.last_time = time.time() - start
                    except Exception as e:
                        st.error(f"Prediction Error: {e}")

        with col2:
            if 'last_res' in st.session_state:
                st.markdown("### Diagnostics")
                
                # Setup variables for HTML
                defect = st.session_state.last_res['predicted_class']
                confidence = st.session_state.last_res['confidence'] * 100
                status = "PASS" if defect in ["Good", "No_Defect"] else "DEFECT"
                color = "#10b981" if status == "PASS" else "#ef4444"
                
                # FIX 2: NO INDENTATION in the HTML string to prevent "Code Block" display
                html_code = f"""
<div class="result-display">
<div style="color: #64748b; letter-spacing: 2px; font-size: 0.8rem;">{status} | DETECTION RESULT</div>
<div class="defect-name">{defect.replace('_', ' ').title()}</div>
<div style="background: #334155; height: 10px; border-radius: 5px; margin-top: 10px;">
<div style="width: {confidence}%; background-color: {color}; height: 100%; border-radius: 5px;"></div>
</div>
<div style="text-align: right; color: #00d9ff; font-weight: bold; margin-top: 5px;">{confidence:.1f}% Confidence</div>
</div>
"""
                st.markdown(html_code, unsafe_allow_html=True)
                
                annotated = visualize_prediction(st.session_state.last_img, st.session_state.last_res)
                # FIX 3: Use old parameter here too
                st.image(annotated, caption="Defect Localization", use_column_width=True)
                
                st.markdown(f"""
<div class="info-card" style="margin-top: 1rem;">
<span class="info-card-label">Inference Latency</span>
<div class="info-card-value">{st.session_state.last_time:.4f}s</div>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
