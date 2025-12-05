import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import os
import json
import time

# ----------------------------------------------------
# Config
# ----------------------------------------------------
st.set_page_config(page_title="CircuitGuard - PCB Defect Detection", layout="wide")

st.title("🔍 CircuitGuard: PCB Defect Detection System")
st.markdown("""
Upload a **PCB image with possible defects**.  
The system automatically picks the correct defect‑free template for that board,
detects defects, groups them into regions, classifies each region, and annotates the PCB.
""")

BASE_DIR = os.path.dirname(__file__)

# ----------------------------------------------------
# Template definitions (adjust names to match your files)
# ----------------------------------------------------
TEMPLATE_FILES = {
    "Board 1":  "template_board1.jpg",
    "Board 4":  "template_board4.jpg",
    "Board 5":  "template_board5.jpg",
    "Board 6":  "template_board6.jpg",
    "Board 7":  "template_board7.jpg",
    "Board 8":  "template_board8.jpg",
    "Board 9":  "template_board9.jpg",
    "Board 10": "template_board10.jpg",
    "Board 11": "template_board11.jpg",
    "Board 12": "template_board12.jpg",
}

def choose_best_template(test_rgb):
    """
    Compare the uploaded image to every template and
    return (best_board_name, best_template_bgr, template_path).
    """
    best_name = None
    best_img_bgr = None
    best_path = None
    best_score = None

    for board_name, fname in TEMPLATE_FILES.items():
        path = os.path.join(BASE_DIR, fname)
        tpl = cv2.imread(path)
        if tpl is None:
            continue

        tpl_rgb = cv2.cvtColor(tpl, cv2.COLOR_BGR2RGB)

        # resize template to match test size
        if tpl_rgb.shape[:2] != test_rgb.shape[:2]:
            tpl_rgb_resized = cv2.resize(tpl_rgb, (test_rgb.shape[1], test_rgb.shape[0]))
        else:
            tpl_rgb_resized = tpl_rgb

        diff = cv2.absdiff(tpl_rgb_resized, test_rgb)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        score = np.mean(gray)  # lower = more similar

        if (best_score is None) or (score < best_score):
            best_score = score
            best_name = board_name
            best_img_bgr = cv2.cvtColor(tpl_rgb_resized, cv2.COLOR_RGB2BGR)
            best_path = path

    return best_name, best_img_bgr, best_path

# ----------------------------------------------------
# Load model (cached)
# ----------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(os.path.join(BASE_DIR, "circuitguard_effnet_model.h5"))

model = load_model()

# ----------------------------------------------------
# Preprocess ROI
# ----------------------------------------------------
def preprocess_roi(roi_bgr):
    img = cv2.resize(roi_bgr, (224, 224))
    img = img.astype("float32") / 255.0
    return img

class_names = [
    "spurious copper",
    "mouse bite",
    "open circuit",
    "missing hole",
    "short",
    "spur"
]

# ----------------------------------------------------
# Upload defect image
# ----------------------------------------------------
st.subheader("📥 Upload PCB Image (Defective)")
file = st.file_uploader("Upload PCB Image", type=["jpg", "jpeg", "png"])

if file:
    img_pil = Image.open(file).convert("RGB")
    st.image(img_pil, caption="Input PCB Image", use_container_width=True)

# ----------------------------------------------------
# Detect Button
# ----------------------------------------------------
if st.button("🚀 Detect Defects", type="primary"):
    if not file:
        st.warning("⚠️ Please upload a PCB image first!")
    else:
        with st.spinner("Processing images..."):

            # 1) Convert uploaded defect image
            test_rgb = np.array(img_pil)
            test_bgr = cv2.cvtColor(test_rgb, cv2.COLOR_RGB2BGR)

            # 2) Automatically choose best template
            best_board, template_bgr, template_path = choose_best_template(test_rgb)
            if template_bgr is None:
                st.error("❌ Could not load any template images. "
                         "Check TEMPLATE_FILES and filenames.")
                st.stop()

            st.info(f"🔎 Detected board design: **{best_board}** "
                    f"(template: `{os.path.basename(template_path)}`)")

            template_rgb = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2RGB)

            # 3) Subtraction
            diff_bgr = cv2.absdiff(template_bgr, test_bgr)
            gray_diff = cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2GRAY)

            # sensitive threshold for mask
            _, mask = cv2.threshold(gray_diff, 5, 255, cv2.THRESH_BINARY)

            # 4) Visual copies (PCB stays green)
            highlighted = test_bgr.copy()
            diff_rgb = cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2RGB)
            mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

            st.success("✅ Processing Complete!")

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("📊 Difference Map")
                st.image(diff_rgb, caption="Detected Differences", use_container_width=True)
            with col_b:
                st.subheader("🎯 Binary Mask")
                st.image(mask_vis, caption="Defect Mask", use_container_width=True)

            # ------------------------------------------------
            # 5) Contours (blobs) with reasonable filter
            # ------------------------------------------------
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            h, w = mask.shape

            min_area = 200          # adjust if needed
            max_area = 0.02 * h * w

            small_boxes = [
                cv2.boundingRect(c)
                for c in contours
                if min_area < cv2.contourArea(c) < max_area
            ]

            st.metric("🔴 Raw defect blobs", len(small_boxes))

            # ------------------------------------------------
            # 6) Merge boxes into regions
            # ------------------------------------------------
            regions = []

            for (x, y, bw, bh) in small_boxes:
                if bh < 15:   # ignore ultra‑thin strips
                    continue

                merged = False
                for i, (rx, ry, rw, rh) in enumerate(regions):
                    if not (x > rx+rw or rx > x+bw or y > ry+rh or ry > y+bh):
                        nx = min(x, rx)
                        ny = min(y, ry)
                        nx2 = max(x + bw, rx + rw)
                        ny2 = max(y + bh, ry + rh)
                        regions[i] = (nx, ny, nx2 - nx, ny2 - ny)
                        merged = True
                        break
                if not merged:
                    regions.append((x, y, bw, bh))

            st.metric("📦 Defect regions (merged)", len(regions))

            if not regions:
                st.info("✅ No significant defects detected after filtering.")
            else:
                # --------------------------------------------
                # 7) Classify and annotate regions
                # --------------------------------------------
                timestamp = int(time.time())
                output_dir = os.path.join(BASE_DIR, f"output_detected_{timestamp}")
                os.makedirs(output_dir, exist_ok=True)

                Image.fromarray(diff_rgb).save(
                    os.path.join(output_dir, "difference_map.png")
                )

                defect_log = []

                st.subheader("🤖 Region‑level Defect Classification")
                roi_cols = st.columns(3)

                overlay = highlighted.copy()
                alpha = 0.35
                display_idx = 0

                for idx, (x, y, bw, bh) in enumerate(regions, start=1):
                    roi_bgr = test_bgr[y:y+bh, x:x+bw]
                    if roi_bgr.size == 0:
                        continue

                    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                    roi_std = np.std(roi_gray)

                    inp = preprocess_roi(roi_bgr)
                    pred = model.predict(np.expand_dims(inp, 0), verbose=0)
                    cls_id = int(np.argmax(pred))
                    conf = float(np.max(pred) * 100.0)
                    label = class_names[cls_id]
                    label_text = f"{label} {conf:.0f}%"

                    defect_log.append({
                        "roi_id": idx,
                        "class_id": cls_id,
                        "class_name": label,
                        "confidence": conf,
                        "bbox": [int(x), int(y), int(bw), int(bh)]
                    })

                    # show only textured ROIs in gallery
                    if roi_std > 8:
                        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
                        with roi_cols[display_idx % 3]:
                            st.image(
                                roi_rgb,
                                caption=f"ROI {idx}: {label_text}",
                                use_container_width=True
                            )
                        display_idx += 1

                        Image.fromarray(roi_rgb).save(
                            os.path.join(output_dir, f"ROI_{idx}.png")
                        )

                    # semi‑transparent red fill
                    cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 0, 255), -1)

                    # thick white outline
                    cv2.rectangle(
                        highlighted,
                        (x, y),
                        (x + bw, y + bh),
                        (255, 255, 255),
                        4,
                        cv2.LINE_AA
                    )

                    # big readable label
                    font_scale = 1.1
                    thickness = 3
                    (text_w, text_h), _ = cv2.getTextSize(
                        label_text,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        thickness
                    )

                    gap = 6
                    label_y_bottom = max(y - gap, text_h + 4)
                    label_y_top = label_y_bottom - text_h - 8

                    cv2.rectangle(
                        highlighted,
                        (x, label_y_top),
                        (x + text_w + 20, label_y_bottom),
                        (0, 0, 0),
                        -1
                    )
                    cv2.putText(
                        highlighted,
                        label_text,
                        (x + 10, label_y_bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA
                    )

                # apply overlay so only regions get red tint
                cv2.addWeighted(overlay, alpha, highlighted, 1 - alpha, 0, highlighted)

                # --------------------------------------------
                # 8) Final annotated PCB
                # --------------------------------------------
                annotated_rgb = cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB)
                st.subheader("📌 Annotated PCB with detected defects")
                st.image(annotated_rgb, use_container_width=True)

                ann_path = os.path.join(output_dir, "annotated_pcb.png")
                Image.fromarray(annotated_rgb).save(ann_path)

                buf_img = io.BytesIO()
                Image.fromarray(annotated_rgb).save(buf_img, format="PNG")
                st.download_button(
                    "⬇️ Download Annotated PCB",
                    data=buf_img.getvalue(),
                    file_name="annotated_pcb.png"
                )

                # --------------------------------------------
                # 9) Defect log
                # --------------------------------------------
                st.subheader("📜 Defect Log")
                st.json(defect_log)

                log_path = os.path.join(output_dir, "defect_log.json")
                with open(log_path, "w") as f:
                    json.dump(defect_log, f, indent=2)

                with open(log_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download Defect Log (JSON)",
                        data=f.read(),
                        file_name="defect_log.json"
                    )

                st.success(f"✅ Results saved in folder: {output_dir}")

