from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import torch
import timm
from torchvision import transforms
from PIL import Image
import base64
import io
import json

app = Flask(__name__)
CORS(app)

# Defect categories predicted by the model
CLASSES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]

MODEL_PATH = "efficientnet_b4_pcb.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
test_transform = None


def load_model():
    """Load EfficientNet-B4 and initialize the preprocessing transform."""
    global model, test_transform

    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Preprocessing settings used during model training
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def classify_roi(roi):
    """Classify a cropped region of interest."""

    # Convert grayscale ROI to RGB for the model
    if len(roi.shape) == 2:
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)

    roi = Image.fromarray(roi)
    tensor = test_transform(roi).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)

    return CLASSES[out.argmax().item()]


def process_images(golden_img, test_img):
    """Core defect detection pipeline using image subtraction and model classification."""

    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    golden_gray = cv2.cvtColor(golden_img, cv2.COLOR_BGR2GRAY)

    # Highlight differences between golden and test PCB
    diff = cv2.absdiff(golden_gray, test_gray)
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    pad = 5
    min_area = 50
    min_size = 10

    img_final = test_img.copy()
    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_size or h < min_size:
            continue

        # Expand the bounding box slightly
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, test_gray.shape[1])
        y2 = min(y + h + pad, test_gray.shape[0])

        roi = test_gray[y1:y2, x1:x2]
        pred = classify_roi(roi)

        detections.append({
            'class': pred,
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'center_x': int(x + w // 2),
            'center_y': int(y + h // 2)
        })

        # Draw bounding box and label
        cv2.rectangle(img_final, (x1, y1), (x2, y2), (0, 0, 255), 2)

        label = pred
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.7
        th = 2

        tx = x1
        ty = max(y1 - 10, 0)
        (tw, tht), base = cv2.getTextSize(label, font, fs, th)

        cv2.rectangle(img_final, (tx, ty - tht - base), (tx + tw, ty + base), (0, 165, 255), -1)
        cv2.putText(img_final, label, (tx, ty), font, fs, (0, 0, 0), th)

    return img_final, detections


@app.route('/')
def index():
    """Serve the HTML frontend."""
    return send_file('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    """Receive golden and test PCB images, run detection, and return results."""
    try:
        data = request.get_json()

        # Extract Base64-encoded image data
        golden_data = data['golden'].split(',')[1]
        test_data = data['test'].split(',')[1]

        golden_bytes = base64.b64decode(golden_data)
        test_bytes = base64.b64decode(test_data)

        golden_arr = np.frombuffer(golden_bytes, np.uint8)
        test_arr = np.frombuffer(test_bytes, np.uint8)

        golden_img = cv2.imdecode(golden_arr, cv2.IMREAD_COLOR)
        test_img = cv2.imdecode(test_arr, cv2.IMREAD_COLOR)

        img_final, detections = process_images(golden_img, test_img)

        # Convert annotated image to Base64 for frontend
        _, buffer = cv2.imencode('.jpg', img_final)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Build CSV text for download
        csv_data = "Defect_ID,Defect_Type,BBox_X1,BBox_Y1,BBox_X2,BBox_Y2,Center_X,Center_Y\n"
        for idx, det in enumerate(detections, 1):
            csv_data += f"{idx},{det['class']},{det['bbox'][0]},{det['bbox'][1]},{det['bbox'][2]},{det['bbox'][3]},{det['center_x']},{det['center_y']}\n"

        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{img_base64}',
            'detections': detections,
            'total': len(detections),
            'csv': csv_data
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("Loading model...")
    load_model()
    print("Model loaded. Server running.")
    app.run(debug=True, port=5000)
