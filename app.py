# -------------------------------
# app.py — Module 05 Web UI
# CircuitGuard PCB Defect Detection
# -------------------------------
import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from pathlib import Path
import pandas as pd

# ==========================
# CONFIG
# ==========================
CKPT_DIR = Path("checkpoints")
OUT_DIR = Path("web_outputs"); OUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224   # Must match your Model

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    ckpt_candidates = list(CKPT_DIR.glob("best_effnet_b0_cpu_ultrafast.pth")) \
                    + list(CKPT_DIR.glob("best_effnet_b0_cpu.pth"))

    if not ckpt_candidates:
        raise FileNotFoundError("No checkpoint found in 'checkpoints/' folder.")

    ckpt = torch.load(ckpt_candidates[0], map_location="cpu")

    label2idx = ckpt["label2idx"]
    idx2label = {v: k for k, v in label2idx.items()}

    model = timm.create_model("efficientnet_b0", pretrained=False,
                              num_classes=len(idx2label))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, idx2label

model, idx2label = load_model()

# ==========================
# IMAGE PIPELINE FUNCTIONS
# ==========================
def preprocess_and_subtract(template, test, resize_to=(1024, 1024)):
    t = cv2.cvtColor(np.array(template), cv2.COLOR_RGB2GRAY)
    s = cv2.cvtColor(np.array(test), cv2.COLOR_RGB2GRAY)

    t = cv2.resize(t, resize_to)
    s = cv2.resize(s, resize_to)

    diff = cv2.absdiff(s, t)
    diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)
    _, th = cv2.threshold(diff_blur, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    return th


def extract_rois(test_img, mask, min_area=80, pad=6):
    img = cv2.cvtColor(np.array(test_img), cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
        crop = img[y0:y1, x0:x1]
        rois.append(((x0, y0, x1, y1), crop))
    return rois


def classify_crop(crop):
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(crop_rgb)

    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])
    t = transform(pil).unsqueeze(0)

    with torch.no_grad():
        logits = model(t)
        probs = F.softmax(logits, dim=1)[0]
        top = int(torch.argmax(probs))
        score = float(probs[top])

    return idx2label[top], score


def annotate_image(test_img, detections):
    img = cv2.cvtColor(np.array(test_img), cv2.COLOR_RGB2BGR)
    out = img.copy()

    for d in detections:
        x0, y0, x1, y1 = d["bbox"]
        label = f"{d['label']} {d['score']:.2f}"

        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(out, label, (x0, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255,255,255), 2)

    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out_rgb)

# ==========================
# STREAMLIT UI
# ==========================
st.title("🔧 CircuitGuard PCB Defect Detection")
st.write("Upload Template + Test images to detect and classify PCB defects.")

template_file = st.file_uploader("📤 Upload Template Image", type=["png","jpg","jpeg"])
test_file = st.file_uploader("📤 Upload Test Image", type=["png","jpg","jpeg"])

if st.button("Run Detection"):

    if not template_file or not test_file:
        st.error("Please upload both images!")
        st.stop()

    template = Image.open(template_file).convert("RGB")
    test = Image.open(test_file).convert("RGB")

    st.image([template, test], caption=["Template", "Test"], width=250)

    st.info("Processing...")
    
    # Subtraction
    mask = preprocess_and_subtract(template, test)

    # ROIs
    rois = extract_rois(test, mask)
    detections = []

    for i, (bbox, crop) in enumerate(rois):
        lbl, score = classify_crop(crop)
        detections.append({
            "roi_idx": i,
            "bbox": bbox,
            "label": lbl,
            "score": score
        })

    # Save CSV
    df = pd.DataFrame(detections)
    csv_path = OUT_DIR / "prediction_log.csv"
    df.to_csv(csv_path, index=False)

    # Annotated image
    annotated = annotate_image(test, detections)

    st.subheader("📌 Annotated Output Image")
    st.image(annotated, width=500)

    st.subheader("📑 Prediction Log")
    st.dataframe(df)

    st.download_button("⬇ Download Prediction CSV",
                       csv_path.read_bytes(),
                       file_name="prediction_log.csv")

    st.success("Processing Complete!")
