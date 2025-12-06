import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torch.nn.functional as F

from .model_loader import load_model

# Load model once
model = load_model("best_efficientnet_b4.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)

# Image preprocessing
transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Class labels
CLASS_NAMES = {
    0: "Short",
    1: "Open",
    2: "Mouse Bite",
    3: "Spur",
    4: "Missing Hole",
    5: "Copper Spatter",
    6: "Pin Hole"
}

CLASS_NAME_TO_ID = {v: k for k, v in CLASS_NAMES.items()}


def predict_defects(template_file, selected_class_name):
    """Detect only the selected defect class from a PCB image."""
    
    TARGET_CLASS_ID = CLASS_NAME_TO_ID[selected_class_name]

    template_file.seek(0)
    img_bytes = np.frombuffer(template_file.read(), np.uint8)
    pcb_img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if pcb_img is None:
        raise ValueError("Image could not be loaded.")

    # Resize image
    H, W = 512, 512
    pcb_img = cv2.resize(pcb_img, (W, H))

    gray = cv2.cvtColor(pcb_img, cv2.COLOR_BGR2GRAY)

    # MSER Detector
    mser = cv2.MSER_create()
    mser.setMinArea(60)
    mser.setMaxArea(5000)

    regions, _ = mser.detectRegions(gray)
    contours = [cv2.convexHull(r.reshape(-1, 1, 2)) for r in regions]

    annotated = pcb_img.copy()
    predictions = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w < 20 or h < 20:
            continue

        roi = pcb_img[y:y+h, x:x+w]
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        roi_tensor = transform(roi_pil).unsqueeze(0).to(DEVICE)

        # Prediction
        with torch.no_grad():
            out = model(roi_tensor)
            probs = F.softmax(out, dim=1)
            cls = torch.argmax(probs).item()
            conf = float(probs[0][cls] * 100)

        # Only keep selected defect
        if cls != TARGET_CLASS_ID:
            continue

        label = f"{CLASS_NAMES[cls]} ({conf:.1f}%)"

        predictions.append({
            "class_id": cls,
            "class_name": CLASS_NAMES[cls],
            "confidence": round(conf, 2),
            "box": [int(x), int(y), int(w), int(h)]
        })

        # Draw box
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return annotated_rgb, predictions
