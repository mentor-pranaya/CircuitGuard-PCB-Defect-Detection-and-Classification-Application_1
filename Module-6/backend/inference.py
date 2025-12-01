import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torch.nn.functional as F

from .model_loader import load_model

# Load classifier model once
model = load_model("best_efficientnet_b4.pth")

# Preprocessing
transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor()
])

# Class names
CLASS_NAMES = {
    0: "Short",
    1: "Open",
    2: "Mouse Bite",
    3: "Spur",
    4: "Missing Hole",
    5: "Copper Spatter",
    6: "Pin Hole"
}


def predict_defects(template_file, test_file):

    # Reset pointer (required for Streamlit)
    template_file.seek(0)
    test_file.seek(0)

    # Read uploaded images
    template_bytes = np.frombuffer(template_file.read(), np.uint8)
    test_bytes = np.frombuffer(test_file.read(), np.uint8)

    template_img = cv2.imdecode(template_bytes, cv2.IMREAD_COLOR)
    test_img = cv2.imdecode(test_bytes, cv2.IMREAD_COLOR)

    if template_img is None:
        raise ValueError("Template image could not be loaded.")

    if test_img is None:
        raise ValueError("Test image could not be loaded.")

    # Resize both to same size
    H, W = 512, 512
    template_img = cv2.resize(template_img, (W, H))
    test_img = cv2.resize(test_img, (W, H))

    # Convert to grayscale
    temp_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # Subtraction
    diff = cv2.absdiff(temp_gray, test_gray)

    # Threshold to create mask
    _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    predictions = []
    annotated = test_img.copy()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Ignore noise
        if w < 15 or h < 15:
            continue

        # Extract ROI
        roi = test_img[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        # Convert ROI → Tensor
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        roi_tensor = transform(roi_pil).unsqueeze(0)

        # Predict
        with torch.no_grad():
            pred = model(roi_tensor)
            cls = torch.argmax(pred).item()

            # Softmax confidence
            probs = F.softmax(pred, dim=1)
            conf = probs[0][cls].item() * 100.0

        # Confidence filter — remove low scores
        if conf < 60:
            continue

        # Clean label
        label = f"{CLASS_NAMES[cls]} {conf:.0f}%"

        # Save result
        predictions.append({
            "class_id": cls,
            "class_name": CLASS_NAMES[cls],
            "confidence": round(conf, 2),
            "box": [x, y, w, h]
        })

        # Draw
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(
            annotated, label, (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 2,
        )

    # Convert for Streamlit
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return annotated_rgb, predictions
