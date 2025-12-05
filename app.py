import streamlit as st
import numpy as np
import cv2
import io
import pandas as pd
import time
import os
from mod6 import full_inference_pipeline, CLASS_NAMES


def convert_cv2_to_bytes(image_array: np.ndarray) -> bytes:

    rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    is_success, buffer = cv2.imencode(".jpg", rgb_image)
    if not is_success:
        raise Exception("Could not encode image to JPEG.")
    
    return io.BytesIO(buffer).getvalue()

# --- STREAMLIT UI ---

st.set_page_config(
    page_title="CircuitGuard: PCB Defect Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ CircuitGuard: Automated PCB Defect Analysis")
st.markdown("Upload a Test PCB image below. The system will **automatically identify the Golden PCB** and proceed with defect analysis.")

# 1. Sidebar for Input
st.sidebar.header("Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload Test PCB Image (.jpg, .png)", 
    type=['jpg', 'jpeg', 'png']
)

# 2. Main Content Area
if uploaded_file is not None:
    
    # Save uploaded file to temporary path for CV2 processing
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Test Image")
        st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)
        
    with col2:
        st.subheader("Processing Status")
        process_button = st.button("🚀 Run CircuitGuard Analysis", type="primary")

        if process_button:
            
            # --- START PIPELINE ---
            start_time = time.time()
            st.info("Starting **Automatic Template Identification** and defect detection...")
            
            try:
                # It returns the annotated image, the results, and the auto-identified ID
                annotated_image_bgr, final_results, template_id_used = full_inference_pipeline(
                    test_image_path=temp_file_path
                )
                
                annotated_image_bytes = convert_cv2_to_bytes(annotated_image_bgr)
                
                end_time = time.time()
                runtime = end_time - start_time
                st.success(f"✅ Analysis Complete! Runtime: **{runtime:.2f} seconds** (Target: $\le 5$s)")
                st.markdown(f"**Auto-Identified Golden PCB ID Used:** `{template_id_used}`")
                
                # 3. Display Results
                st.subheader("Annotated Result")
                st.image(annotated_image_bytes, caption="Detected Defects", use_column_width=True)

                # 4. Results Log
                if final_results:
                    st.subheader("Classification Log")
                    
                    # Convert list of dicts to DataFrame for display
                    df_log = pd.DataFrame([
                        {'Defect Type': res['prediction'], 
                         'Confidence (%)': f"{res['confidence']:.2f}",
                         'BBox (x,y,w,h)': str(res['bbox'])}
                        for res in final_results
                    ])
                    st.dataframe(df_log, use_container_width=True)
                else:
                    st.info("No defects detected on this PCB.")
                
                # 5. Export Functionality 
                st.download_button(
                    label="⬇️ Download Annotated Image (.jpg)",
                    data=annotated_image_bytes,
                    file_name=f"CircuitGuard_Result_{os.path.basename(uploaded_file.name)}",
                    mime="image/jpeg"
                )
                
            except Exception as e:
                st.error(f"An error occurred during pipeline execution: {e}")
                st.warning("Please check your configuration constants (paths, model weights, etc.) in `backend_pipeline.py`.")
            
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        else:
            st.warning("Click 'Run CircuitGuard Analysis' to start detection.")

else:
    st.info("Please upload a Test PCB image to begin analysis.")