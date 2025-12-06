import streamlit as st
import numpy as np
import cv2
from inference import load_model, process_uploaded_defect_image

st.title("PCB Defect Detection & Visualization")

MODEL_PATH = r"C:\Users\Kandu\OneDrive\Desktop\PCB_DATASET\efficientnet_b4_pcb.pth"
TEMPLATE_DIR = r"C:\Users\Kandu\OneDrive\Desktop\PCB_DATASET\PCB_USED"

@st.cache_resource
def load_effnet_model():
    return load_model(MODEL_PATH)

model = load_effnet_model()

uploaded_file = st.file_uploader("Upload Defect PCB Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    defect_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(defect_img, cv2.COLOR_BGR2RGB), caption="Uploaded Defect Image", use_column_width=True)

    with st.spinner("Processing defect image..."):
        try:
            results = process_uploaded_defect_image(defect_img, TEMPLATE_DIR, model)

            st.subheader("Matched Defect-Free Template Image")ś
            st.image(cv2.cvtColor(results["template_img"], cv2.COLOR_BGR2RGB), caption=results["template_path"], use_column_width=True)

            st.subheader("Aligned Defect Image (Grayscale)")
            st.image(results["aligned_defect_gray"], caption="Aligned Defect Image", use_column_width=True)

            st.subheader("Difference Map (Absolute Difference)")
            st.image(results["diff_img"], caption="Difference Map", use_column_width=True)

            st.subheader("Defect Mask (Thresholded)")
            st.image(results["defect_mask"], caption="Defect Mask", use_column_width=True)

            st.subheader("Annotated Defect Image with Bounding Boxes and Labels")
            st.image(cv2.cvtColor(results["annotated_img"], cv2.COLOR_BGR2RGB), caption="Annotated Defect Image", use_column_width=True)

            if results["rois"]:
                st.subheader("Cropped Defect ROIs with Labels")
                for i, (label, roi_img) in enumerate(results["rois"]):
                    st.image(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB), caption=f"ROI {i+1}: {label}", width=150)

                st.subheader("Defect Detection Log")
                for i, d in enumerate(results["defect_log"]):
                    st.write(f"{i+1}. Label: {d['label']}, Confidence: {d['confidence']:.2f}, Box: ({d['x']}, {d['y']}, {d['w']}, {d['h']})")
            else:
                st.write("No defects detected.")

        except Exception as e:
            st.error(f"Error: {e}")
