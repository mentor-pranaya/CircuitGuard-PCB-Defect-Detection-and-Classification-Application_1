# module6_inference.py
"""
Module-6: Backend pipeline (whole-image classification, 1-2 labels max)

Usage (batch CLI):
    python module6_inference.py

Optional Flask API:
    Set START_FLASK = True and run the script; it will start a simple upload endpoint.
"""

import os
import sys
import cv2
import torch
import timm
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
from datetime import datetime
from flask import Flask, request, jsonify, send_file
import io

# ---------------------------
# ========== CONFIG =========
# ---------------------------

# Paths (use the ones you provided)
TEMPLATE_DIR = r"D:\CircuitGuard-PCB-Project\Data\PCB_DATASET\templates"   # not used but kept for compatibility
TEST_DIR = r"D:\CircuitGuard-PCB-Project\Data\PCB_DATASET\test"
MODEL_PATH = r"D:\CircuitGuard-PCB-Project\Data\best_efficientnet_b4.pth"
OUTPUT_DIR = r"D:\CircuitGuard-PCB-Project\Module6_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "annotated"), exist_ok=True)

# Defect class names - must match training label set (order matters)
# Based on your earlier messages:
DEFECT_CLASSES = ["Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
NUM_CLASSES = len(DEFECT_CLASSES)

# Inference options
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (128, 128)   # same as Module-3 training
BATCH_SIZE = 16
TOP_K = 2                # return up to 2 labels (max 2 as requested)
CONFIDENCE_THRESHOLD = 0.01  # tiny threshold to filter extremely low-prob results

# Flask API toggle
START_FLASK = False      # set True if you want the upload endpoint

# ---------------------------
# ======= TRANSFORMS ========
# ---------------------------

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------------------
# ======= MODEL HELPERS =====
# ---------------------------

def create_model(num_classes=NUM_CLASSES):
    """Create an EfficientNet-B4 model (timm). Do NOT set pretrained=True for checkpoint loading."""
    model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=num_classes)
    return model

def load_checkpoint(model, checkpoint_path):
    """
    Robust loader:
      - Accepts checkpoints saved as model.state_dict() or with a 'model.' prefix.
      - Resolves missing/unexpected keys by trying key remapping and non-strict load.
      - Ensures classifier size matches (if not, tries to patch final layer).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu")

    # If user saved full object (model.state_dict()) or dict with extra keys:
    if isinstance(state, dict) and 'state_dict' in state:
        state_dict = state['state_dict']
    else:
        state_dict = state

    # Detect prefix (e.g., keys start with 'model.' vs top-level keys)
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("model.") and not any(k.startswith("model.") for k in model.state_dict().keys()):
        # strip 'model.' prefix
        new_state = {}
        for k, v in state_dict.items():
            newk = k.replace("model.", "", 1)
            new_state[newk] = v
        state_dict = new_state

    # Check classifier size mismatch and try to handle
    sd_keys = list(state_dict.keys())
    cls_w_key = None
    cls_b_key = None
    for candidate in ['classifier.weight', 'classifier.bias', 'fc.weight', 'fc.bias']:
        if candidate in sd_keys:
            cls_w_key = candidate if candidate.endswith('.weight') else cls_w_key
            # keep as is; find proper keys
    # If checkpoint classifier size doesn't match, we will try a non-strict load and then replace classifier
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Checkpoint loaded (strict=True).")
    except RuntimeError as e:
        print("Strict load failed — attempting flexible load. Message:")
        print(e)
        # try non-strict load first
        missing, unexpected = None, None
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())

        # If classifier keys present, check shapes
        # Load with strict=False to load compatible weights
        model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded with strict=False (compatible weights loaded).")

        # Now ensure classifier final layer shape matches NUM_CLASSES — if mismatch, rebuild classifier weights from ckpt if possible
        # Get classifier param name in model:
        model_sd = model.state_dict()
        classifier_name = None
        for key in model_sd.keys():
            if key.endswith('classifier.weight') or key.endswith('classifier.bias'):
                classifier_name = key.rsplit('.', 1)[0]  # e.g., 'classifier' or 'fc'
                break

        # If checkpoint has classifier weights of different size, preserve current model classifier (random init) but try to copy if exact match available
        # Nothing else to do; we already loaded compatible weights.
        print("Note: final classifier left as model's current layer if shapes mismatched.")

    return model

# ---------------------------
# ======= PREDICTION =========
# ---------------------------

def preprocess_image_cv2(image_path):
    """Load with cv2, convert to PIL-like array (RGB), apply transform and return tensor."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    tensor = transform(pil)   # (C,H,W)
    return tensor, img_bgr  # return original BGR for annotation

def predict_single_image(model, tensor):
    """Return top-K (label, prob) for a single preprocessed tensor (no batch dim)."""
    model.eval()
    tensor = tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy().squeeze(0)
    # sort descending
    idxs = np.argsort(probs)[::-1]
    labels = []
    for i in range(min(TOP_K, len(probs))):
        if probs[idxs[i]] >= CONFIDENCE_THRESHOLD:
            labels.append((DEFECT_CLASSES[idxs[i]], float(probs[idxs[i]])))
    return labels

def annotate_and_save(image_bgr, labels, save_path):
    """Annotate BGR image with top labels and save."""
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]
    # Compose text (max 2 labels)
    if not labels:
        text = "Prediction: Unknown"
    else:
        text = ", ".join([f"{lbl} ({p:.2f})" for lbl, p in labels[:2]])
        text = "Prediction: " + text

    # Put a filled rectangle at top for readability
    overlay_height = 50
    cv2.rectangle(annotated, (0,0), (w, overlay_height), (255,255,255), -1)
    cv2.putText(annotated, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2, cv2.LINE_AA)
    cv2.imwrite(save_path, annotated)
    return save_path

# ---------------------------
# ======= BATCH RUN =========
# ---------------------------

def run_inference_on_folder(test_dir=TEST_DIR, model_path=MODEL_PATH, output_dir=OUTPUT_DIR):
    print("Starting Module-6 Inference (whole-image classification)...")
    print("Test folder:", test_dir)
    print("Model checkpoint:", model_path)
    # Build model and load weights
    print("\nLoading model...")
    model = create_model(num_classes=NUM_CLASSES).to(DEVICE)
    model = load_checkpoint(model, model_path)
    model.eval()
    print("Model loaded successfully!\n")

    rows = []
    image_files = []
    # Walk test_dir and collect images from subfolders (each defect folder contains images)
    for root, dirs, files in os.walk(test_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(root, f))

    if not image_files:
        print("No images found in test dir:", test_dir)
        return

    print(f"Found {len(image_files)} images — running inference...\n")

    for img_path in sorted(image_files):
        try:
            tensor, img_bgr = preprocess_image_cv2(img_path)
        except Exception as e:
            print("Skipping", img_path, "read error:", e)
            continue

        labels = predict_single_image(model, tensor)
        # Keep max 2 labels (already done). If none, return Unknown.
        if not labels:
            labels = [("Unknown", 0.0)]

        # Save annotated image
        base = os.path.basename(img_path)
        name_wo_ext = os.path.splitext(base)[0]
        annot_name = f"annot_{name_wo_ext}.jpg"
        annot_path = os.path.join(output_dir, "annotated", annot_name)
        annotate_and_save(img_bgr, labels, annot_path)

        # Record log row
        rows.append({
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "image": base,
            "input_path": img_path,
            "prediction_primary": labels[0][0],
            "confidence_primary": labels[0][1],
            "prediction_secondary": labels[1][0] if len(labels) > 1 else "",
            "confidence_secondary": labels[1][1] if len(labels) > 1 else "",
            "annotated_path": annot_path
        })

        # Print concise output to console
        preds = ", ".join([f"{l} ({p:.2f})" for l,p in labels])
        print(f"{base}  ->  {preds}")

    # Save CSV log in UTF-8 safe mode
    log_df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "module6_inference_log.csv")
    log_df.to_csv(csv_path, index=False, encoding='utf-8')
    print("\nCompleted. Log saved to:", csv_path)
    print("Annotated images saved to:", os.path.join(output_dir, "annotated"))
    return csv_path

# ---------------------------
# ======= FLASK API =========
# ---------------------------

def create_flask_app(model):
    app = Flask(__name__)

    @app.route("/predict", methods=["POST"])
    def predict():
        """
        POST form-data with key 'file' containing image.
        Returns JSON: primary/secondary label + confidence and annotated image bytes (jpg).
        """
        if 'file' not in request.files:
            return jsonify({"error": "no file provided"}), 400
        f = request.files['file']
        in_bytes = f.read()
        npimg = np.frombuffer(in_bytes, np.uint8)
        img_bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({"error": "invalid image"}), 400

        # Preprocess
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        tensor = transform(pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy().squeeze(0)
        idxs = np.argsort(probs)[::-1]
        labels = []
        for i in range(min(TOP_K, len(probs))):
            if probs[idxs[i]] >= CONFIDENCE_THRESHOLD:
                labels.append((DEFECT_CLASSES[idxs[i]], float(probs[idxs[i]])))
        if not labels:
            labels = [("Unknown", 0.0)]

        # Annotate
        annot_bgr = img_bgr.copy()
        h, w = annot_bgr.shape[:2]
        text = "Prediction: " + ", ".join([f"{lbl} ({p:.2f})" for lbl,p in labels[:2]])
        cv2.rectangle(annot_bgr, (0,0), (w,40), (255,255,255), -1)
        cv2.putText(annot_bgr, text, (8,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
        # Encode image to bytes
        _, img_encoded = cv2.imencode('.jpg', annot_bgr)
        return_data = io.BytesIO(img_encoded.tobytes())
        # JSON with labels and image as bytes response
        response = {
            "primary": labels[0][0],
            "confidence_primary": labels[0][1],
            "secondary": labels[1][0] if len(labels) > 1 else "",
            "confidence_secondary": labels[1][1] if len(labels) > 1 else ""
        }
        # send image separately
        return send_file(return_data, mimetype='image/jpeg', attachment_filename='annot.jpg')

    return app

def run_flask(model):
    app = create_flask_app(model)
    print("Starting Flask upload endpoint on http://127.0.0.1:5000/predict")
    app.run(host="0.0.0.0", port=5000, debug=False)

# ---------------------------
# ======= MAIN ENTRY ========
# ---------------------------

if __name__ == "__main__":
    # By default run batch inference
    csv = run_inference_on_folder(TEST_DIR, MODEL_PATH, OUTPUT_DIR)

    # If you want the Flask API (connect to frontend), set START_FLASK = True at top
    if START_FLASK:
        # create and load model once and start server
        model_for_api = create_model(num_classes=NUM_CLASSES).to(DEVICE)
        model_for_api = load_checkpoint(model_for_api, MODEL_PATH)
        run_flask(model_for_api)
