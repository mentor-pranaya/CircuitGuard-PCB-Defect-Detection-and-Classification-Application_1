import streamlit as st
import numpy as np
import cv2
import os
import pandas as pd
from datetime import datetime

from inference import load_model, process_uploaded_defect_image

st.title("PCB Defect Detection & Visualization")

MODEL_PATH = r"C:\Users\Kandu\OneDrive\Desktop\PCB_DATASET\efficientnet_b4_pcb.pth"
TEMPLATE_DIR = r"C:\Users\Kandu\OneDrive\Desktop\PCB_DATASET\PCB_USED"

# ================================
# MODULE 7: OUTPUT DIRECTORIES
# ================================
OUTPUT_DIR = "module7_outputs"
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated_images")
LOG_PATH = os.path.join(OUTPUT_DIR, "detection_log.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)


# ================================
# LOAD MODEL
# ================================
@st.cache_resource
def load_effnet_model():
    return load_model(MODEL_PATH)

model = load_effnet_model()


# ================================
# FILE UPLOAD
# ================================
uploaded_file = st.file_uploader("Upload Defect PCB Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    defect_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(defect_img, cv2.COLOR_BGR2RGB),
             caption="Uploaded Defect Image",
             use_column_width=True)

    with st.spinner("Processing defect image..."):
        try:
            results = process_uploaded_defect_image(
                defect_img, TEMPLATE_DIR, model
            )

            # ================================
            # SHOW TEMPLATE IMAGE
            # ================================
            st.subheader("Matched Defect-Free Template Image")
            st.image(cv2.cvtColor(results["template_img"], cv2.COLOR_BGR2RGB),
                     caption=results["template_path"],
                     use_column_width=True)

            # ================================
            # VISUALIZATION STEPS
            # ================================
            st.subheader("Aligned Defect Image (Grayscale)")
            st.image(results["aligned_defect_gray"],
                     caption="Aligned Defect Image",
                     use_column_width=True)

            st.subheader("Difference Map (Absolute Difference)")
            st.image(results["diff_img"], caption="Difference Map",
                     use_column_width=True)

            st.subheader("Defect Mask (Thresholded)")
            st.image(results["defect_mask"], caption="Defect Mask",
                     use_column_width=True)

            # ================================
            # ANNOTATED IMAGE
            # ================================
            st.subheader("Annotated Defect Image with Bounding Boxes and Labels")
            annotated_rgb = cv2.cvtColor(results["annotated_img"], cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Annotated Defect Image",
                     use_column_width=True)

            # ================================
            # SAVE ANNOTATED IMAGE
            # ================================
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            annotated_filename = f"annotated_{timestamp}.png"
            annotated_path = os.path.join(ANNOTATED_DIR, annotated_filename)

            cv2.imwrite(annotated_path, results["annotated_img"])

            # ================================
            # SHOW ROIs
            # ================================
            if results["rois"]:
                st.subheader("Cropped Defect ROIs with Labels")
                for i, (label, roi_img) in enumerate(results["rois"]):
                    st.image(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB),
                             caption=f"ROI {i+1}: {label}",
                             width=150)

                # ================================
                # DEFECT LOG DISPLAY
                # ================================
                st.subheader("Defect Detection Log")
                for i, d in enumerate(results["defect_log"]):
                    st.write(f"{i+1}. Label: {d['label']}, "
                             f"Confidence: {d['confidence']:.2f}, "
                             f"Box: ({d['x']}, {d['y']}, {d['w']}, {d['h']})")

                # ================================
                # CREATE / APPEND CSV LOG
                # ================================
                df_log = pd.DataFrame(results["defect_log"])
                df_log["timestamp"] = timestamp
                df_log["input_image"] = uploaded_file.name
                df_log["template_used"] = results["template_path"]

                if os.path.exists(LOG_PATH):
                    df_log.to_csv(LOG_PATH, mode="a", header=False, index=False)
                else:
                    df_log.to_csv(LOG_PATH, index=False)

                # ================================
                # DOWNLOAD BUTTONS
                # ================================
                st.subheader("Download Results")

                # --- Download annotated image ---
                with open(annotated_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Annotated Image",
                        data=f.read(),
                        file_name=annotated_filename,
                        mime="image/png"
                    )

                # --- Download CSV log ---
                with open(LOG_PATH, "rb") as f:
                    st.download_button(
                        label="📥 Download Defect Log (CSV)",
                        data=f.read(),
                        file_name="pcb_defect_log.csv",
                        mime="text/csv"
                    )

            else:
                st.write("No defects detected.")

        except Exception as e:
            st.error(f"Error: {e}")
