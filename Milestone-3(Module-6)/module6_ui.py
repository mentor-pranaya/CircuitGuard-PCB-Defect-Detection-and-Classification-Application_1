import streamlit as st
import torch
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
import cv2
import os
import pandas as pd

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = r"D:\CircuitGuard-PCB-Project\Data\best_efficientnet_b4.pth"
OUTPUT_DIR = r"D:\CircuitGuard-PCB-Project\Module6_outputs"
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)

CLASS_NAMES = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper"
]

TARGET_SIZE = (128, 128)


# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=6)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# -------------------------------
# PREPROCESSING
# -------------------------------
def preprocess_image(img_pil):
    img = img_pil.resize(TARGET_SIZE)
    img = np.array(img).astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, (2, 0, 1))
    
    # FIX: ensure float32 for PyTorch
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return img_tensor



# -------------------------------
# ANNOTATION
# -------------------------------
def annotate_image(img_pil, label1, conf1, label2=None, conf2=None):
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    text = f"{label1} ({conf1:.2f})"
    if label2:
        text += f" | {label2} ({conf2:.2f})"

    cv2.putText(img, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# -------------------------------
# STREAMLIT UI
# -------------------------------
def main():
    st.title("🔍 PCB DEFECT DETECTION — Module-6 UI")
    st.markdown("Upload a PCB image to classify defects using EfficientNet-B4")

    model = load_model()

    uploaded_file = st.file_uploader("Upload PCB Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # Show user image
        img_pil = Image.open(uploaded_file).convert("RGB")
        st.image(img_pil, caption="Uploaded Image", use_column_width=True)

        # Run inference
        if st.button("Run Inference"):
            with st.spinner("Processing..."):
                img_tensor = preprocess_image(img_pil)
                outputs = model(img_tensor)
                probs = F.softmax(outputs, dim=1).detach().numpy()[0]

                # Top-2 predictions
                idx = np.argsort(probs)[::-1]
                top1 = idx[0]
                top2 = idx[1]

                label1, conf1 = CLASS_NAMES[top1], probs[top1]
                label2, conf2 = CLASS_NAMES[top2], probs[top2]

                # Annotate & save
                annotated = annotate_image(img_pil, label1, conf1, label2, conf2)

                save_path = os.path.join(ANNOTATED_DIR, f"annotated_{uploaded_file.name}")
                annotated.save(save_path)

                # Log entry
                log_path = os.path.join(OUTPUT_DIR, "module6_inference_log.csv")

                df_entry = pd.DataFrame([{
                    "filename": uploaded_file.name,
                    "top1_class": label1,
                    "top1_conf": round(float(conf1), 4),
                    "top2_class": label2,
                    "top2_conf": round(float(conf2), 4)
                }])

                if os.path.exists(log_path):
                    df_entry.to_csv(log_path, mode="a", header=False, index=False)
                else:
                    df_entry.to_csv(log_path, index=False)

            # Show Results
            st.success("Inference Completed!")

            st.subheader("Prediction Result")
            st.write(f"### 🥇 **Top-1:** {label1} — ({conf1:.2f})")
            st.write(f"### 🥈 **Top-2:** {label2} — ({conf2:.2f})")

            st.subheader("Annotated Output")
            st.image(annotated, use_column_width=True)

            # Download button for annotated image
            st.download_button(
                label="Download Annotated Image",
                data=open(save_path, "rb").read(),
                file_name=f"annotated_{uploaded_file.name}",
                mime="image/jpeg"
            )

            # Download log
            st.download_button(
                label="Download Full Log (CSV)",
                data=open(log_path, "rb").read(),
                file_name="module6_inference_log.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
