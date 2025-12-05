import os
import io
import time
from typing import List, Tuple

import streamlit as st
from PIL import Image
import cv2
import numpy as np

from utils.data_loader import load_imagefolder_paths, make_dataloaders, TARGET_SIZE
from utils.preprocess import pil_to_cv2, cv2_to_pil, detect_diffs, annotate_image

from utils.file_utils import ensure_dir
from models.classifier import create_model, load_model_checkpoint, predict_patch
from models.trainer import train_loop
from models.evaluator import evaluate_on_folder

st.set_page_config(page_title="PCB Defect Suite", layout="wide", initial_sidebar_state="expanded")
ensure_dir("outputs")
ensure_dir("outputs/trained_models")
ensure_dir("outputs/eval")

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

st.title("PCB Defect Detection & Classification (Modular)")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    rois_dir = st.text_input("ROI dataset folder (ImageFolder)", value="dataset")
    model_path = st.text_input("Model save/load path", value=os.path.join("outputs", "trained_models", "efficientnet_b0.pth"))
    model_name = st.text_input("Timm model name", value="efficientnet_b0")
    epochs = st.number_input("Epochs", min_value=1, value=10)
    batch_size = st.number_input("Batch size", min_value=1, value=16)
    lr = st.number_input("Learning rate", value=1e-4, format="%.6f")
    st.write(f"Device: **{DEVICE}**")

tabs = st.tabs(["Prep", "Train", "Evaluate", "Inference"])

# -----------------------------
# Prep (data inspection + ROI)
# -----------------------------
with tabs[0]:
    st.header("Prep — dataset inspection & ROI extraction")
    files, labels = load_imagefolder_paths(rois_dir)
    st.write(f"Found {len(files)} files in `{rois_dir}`" if files else f"No files found in `{rois_dir}`")
    col1, col2 = st.columns([1,2])
    with col1:
        up_tmpl = st.file_uploader("Template (golden) image", type=["png","jpg","jpeg"], key="prep_tmpl")
        up_test = st.file_uploader("Test image", type=["png","jpg","jpeg"], key="prep_test")
        min_area = st.number_input("Min ROI area (px)", value=50, min_value=10)
        blur_k = st.slider("Gaussian blur kernel (odd)", 1, 21, 5, step=2)
        thresh_user = st.slider("User threshold (0-255)", 1, 255, 40)
        run_extract = st.button("Run subtraction & extract ROIs")
    with col2:
        preview = st.empty()

    if run_extract:
        if not up_tmpl or not up_test:
            st.error("Upload both template and test images.")
        else:
            tpl = Image.open(up_tmpl).convert("RGB")
            tst = Image.open(up_test).convert("RGB")
            tpl_cv2, tst_cv2 = pil_to_cv2(tpl), pil_to_cv2(tst)
            boxes, mask = detect_diffs(tpl_cv2, tst_cv2, blur=blur_k, thresh_val=thresh_user, min_area=min_area)
            ann = annotate_image(tst_cv2, boxes, ["?"]*len(boxes), [0.0]*len(boxes))
            preview.subheader("Annotated differences")
            st.image(cv2_to_pil(ann), use_column_width=True)
            preview.subheader("Mask")
            st.image(Image.fromarray(mask), use_column_width=True)
            if boxes:
                st.subheader(f"Extracted {len(boxes)} ROIs")
                cols = st.columns(min(6, len(boxes)))
                for i,(x,y,w,h) in enumerate(boxes):
                    patch = tst_cv2[y:y+h, x:x+w]
                    with cols[i % len(cols)]:
                        st.image(cv2_to_pil(patch), caption=f"ROI {i} ({w}x{h})", use_column_width=True)
                if st.button("Save extracted ROIs to ./extracted_rois"):
                    out_dir = "extracted_rois"
                    ensure_dir(out_dir)
                    for i,(x,y,w,h) in enumerate(boxes):
                        patch = tst_cv2[y:y+h, x:x+w]
                        fname = os.path.join(out_dir, f"roi_{i}.png")
                        cv2.imwrite(fname, cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
                    st.success(f"Saved {len(boxes)} ROIs to `{out_dir}`")

# -----------------------------
# Train
# -----------------------------
with tabs[1]:
    st.header("Train model")
    st.write("Dataset should be ImageFolder (train/<class>/*.png, test/<class>/*.png)")
    start_train = st.button("Start training")
    if start_train:
        files_all, labels_all = load_imagefolder_paths(rois_dir)
        if not files_all:
            st.error(f"No dataset found in `{rois_dir}`")
        else:
            num_classes = len(sorted(set(labels_all)))
            st.info(f"Classes detected: {sorted(set(labels_all))}  —  Num classes: {num_classes}")
            model = create_model(model_name=model_name, num_classes=num_classes, pretrained=True)
            history, best_path = train_loop(
                model=model,
                rois_dir=rois_dir,
                model_save_path=model_path,
                epochs=int(epochs),
                batch_size=int(batch_size),
                lr=float(lr),
                device=DEVICE
            )
            st.success(f"Training complete. Best model saved to {best_path}")
            # plot losses/acc (simple)
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1,2, figsize=(10,4))
                ax[0].plot(history["train_loss"], label="train_loss"); ax[0].plot(history["val_loss"], label="val_loss"); ax[0].legend(); ax[0].set_title("Loss")
                ax[1].plot(history["val_acc"], label="val_acc"); ax[1].legend(); ax[1].set_title("Val acc (%)")
                st.pyplot(fig)
            except Exception:
                pass

# -----------------------------
# Evaluate
# -----------------------------
with tabs[2]:
    st.header("Evaluate model on ImageFolder (test set)")
    eval_model_path = st.text_input("Model path to evaluate", value=model_path)
    test_folder = st.text_input("Test folder (ImageFolder)", value=os.path.join(rois_dir, "test"))
    run_eval = st.button("Run evaluation")
    if run_eval:
        if not os.path.exists(test_folder):
            st.error(f"Test folder not found: {test_folder}")
        elif not os.path.exists(eval_model_path):
            st.error(f"Model not found: {eval_model_path}")
        else:
            report, cm_fig, out_dir = evaluate_on_folder(eval_model_path, test_folder, model_name=model_name, device=DEVICE)
            st.subheader("Classification report")
            st.text(report)
            st.subheader("Confusion matrix")
            st.pyplot(cm_fig)
            st.success(f"Annotated visuals saved to: {out_dir}")

# -----------------------------
# Inference
# -----------------------------
with tabs[3]:
    st.header("Inference (template subtraction + classification)")
    tpl_file = st.file_uploader("Template (golden)", type=["png","jpg","jpeg"], key="inf_tpl")
    test_file = st.file_uploader("Test image", type=["png","jpg","jpeg"], key="inf_test")
    infer_model_path = st.text_input("Model checkpoint for inference", value=model_path, key="inf_model")
    blur_k = st.slider("Blur (odd)", 1, 21, 5, step=2)
    thresh_val = st.slider("User threshold", 1, 255, 40)
    min_area = st.number_input("Min ROI area", value=50, min_value=10)
    run_infer = st.button("Run inference")
    display = st.empty()

    if run_infer:
        if not tpl_file or not test_file:
            st.error("Upload both template and test images.")
        elif not os.path.exists(infer_model_path):
            st.error(f"Model checkpoint not found: {infer_model_path}")
        else:
            tpl_img = Image.open(tpl_file).convert("RGB")
            test_img = Image.open(test_file).convert("RGB")
            tpl_cv2 = pil_to_cv2(tpl_img)
            test_cv2 = pil_to_cv2(test_img)
            boxes, mask = detect_diffs(tpl_cv2, test_cv2, blur=blur_k, thresh_val=thresh_val, min_area=min_area)

            # class names inference from dataset
            _, labels_try = load_imagefolder_paths(rois_dir)
            class_names = sorted(set(labels_try)) if labels_try else [f"class_{i}" for i in range(6)]

            model_inf = load_model_checkpoint(infer_model_path, model_name=model_name, num_classes=len(class_names), device=DEVICE)

            labels_pred, confs = [], []
            patches = []
            for (x,y,w,h) in boxes:
                patch = test_cv2[y:y+h, x:x+w]
                if patch.size == 0:
                    continue
                pil_patch = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
                idx, conf = predict_patch(model_inf, pil_patch, device=DEVICE)
                label = class_names[idx] if idx < len(class_names) else f"cls{idx}"
                labels_pred.append(label)
                confs.append(conf)
                patches.append((pil_patch, label, conf))

            annotated = annotate_image(test_cv2, boxes, labels_pred, confs)
            display.subheader("Annotated result")
            st.image(cv2_to_pil(annotated), use_column_width=True)
            display.subheader("Diff mask")
            st.image(Image.fromarray(mask), use_column_width=True)

            if patches:
                st.subheader("Detected patches")
                cols = st.columns(min(6, len(patches)))
                for i, (p,l,c) in enumerate(patches):
                    with cols[i % len(cols)]:
                        st.image(p, caption=f"{l} ({c:.2f})", use_column_width=True)

            # download annotated
            buf = io.BytesIO()
            cv2_to_pil(annotated).save(buf, format="PNG")
            buf.seek(0)
            st.download_button("Download annotated image", data=buf, file_name="annotated.png", mime="image/png")

st.markdown("---")
st.markdown("Notes: Use GPU for heavy training. Streamlit runs in a single process; for long training prefer separate script.")
