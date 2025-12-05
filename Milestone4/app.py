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
import os

app = Flask(__name__)
CORS(app)

# The six PCB defect classes your classifier was trained on
CLASSES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]

MODEL_PATH = "efficientnet_b4_pcb.pth"

# IMPORTANT:
# Place all your golden PCB images inside a folder named "golden"
# (Same directory as this Flask script)
GOLDEN_FOLDER = "golden"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
test_transform = None
golden_pcbs = []


def load_model():
    """Loads the EfficientNet model used for defect classification."""
    global model, test_transform

    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Image transform applied to each ROI before classification
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def load_golden_pcbs():
    """Loads every golden PCB image from the golden/ folder.

    The folder MUST exist and contain all PCB reference images.
    """
    global golden_pcbs
    golden_pcbs = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    golden_path = os.path.join(script_dir, GOLDEN_FOLDER)

    if os.path.exists(golden_path):
        for filename in os.listdir(golden_path):
            # Only accept image files
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(golden_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is not None:
                    golden_pcbs.append((filename, img))

        print(f"Loaded {len(golden_pcbs)} golden PCB images")
    else:
        print("Golden folder not found. Please create a folder named 'golden' and add golden PCB images.")


def identify_golden_pcb(test_img_gray, golden_pcbs, nfeatures=2000):
    """Automatically selects the best matching golden PCB using ORB feature matching.

    Returns:
        best_img   → the matched golden image
        best_name  → filename of the matched golden PCB
        best_matches → number of ORB matches (higher = better)
    """

    if not golden_pcbs:
        return None, None, 0

    orb = cv2.ORB_create(nfeatures=nfeatures)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    kp_test, des_test = orb.detectAndCompute(test_img_gray, None)
    if des_test is None:
        return None, None, 0

    best_name = None
    best_matches = 0
    best_img = None

    # Compare the test PCB with every golden PCB
    for name, golden_img in golden_pcbs:
        kp_golden, des_golden = orb.detectAndCompute(golden_img, None)
        if des_golden is None:
            continue

        matches = bf.match(des_test, des_golden)
        score = len(matches)

        # Keep whichever golden PCB gives the highest match count
        if score > best_matches:
            best_matches = score
            best_name = name
            best_img = golden_img

    return best_img, best_name, best_matches


def classify_roi(roi):
    """Runs the CNN classifier on a cropped defect ROI and returns label + confidence."""

    # Convert grayscale → RGB (model expects 3 channels)
    if len(roi.shape) == 2:
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)

    roi = Image.fromarray(roi)
    t = test_transform(roi).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(t)
        probabilities = torch.nn.functional.softmax(out, dim=1)
        confidence = probabilities.max().item()
        pred_idx = out.argmax().item()

    return CLASSES[pred_idx], confidence


def process_images(golden_img, test_img):
    """Subtracts test PCB from golden PCB, finds defects, classifies them, and annotates output."""

    # Convert to grayscale for subtraction
    img_BW = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    golden_BW = cv2.cvtColor(golden_img, cv2.COLOR_BGR2GRAY)

    # Pixel-wise subtraction highlights defect regions
    sub = cv2.absdiff(golden_BW, img_BW)
    _, thresh = cv2.threshold(sub, 20, 255, cv2.THRESH_BINARY)

    # Locate defect contours
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

        # Slightly expand bounding box
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, img_BW.shape[1])
        y2 = min(y + h + pad, img_BW.shape[0])

        roi = img_BW[y1:y2, x1:x2]

        pred, confidence = classify_roi(roi)

        detections.append({
            'class': pred,
            'confidence': round(confidence * 100, 2),
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'center_x': int(x + w // 2),
            'center_y': int(y + h // 2)
        })

        # Draw bounding box + label
        cv2.rectangle(img_final, (x1, y1), (x2, y2), (0, 0, 255), 2)

        label = pred
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.7
        th = 2
        tx = x1
        ty = max(y1 - 10, 0)

        (tw, tht), base = cv2.getTextSize(label, font, fs, th)

        # Background box for label
        cv2.rectangle(img_final,
                      (tx, ty - tht - base),
                      (tx + tw, ty + base),
                      (0, 165, 255),
                      -1)

        cv2.putText(img_final, label, (tx, ty),
                    font, fs, (0, 0, 0), th)

    # Sort defects by confidence (highest first)
    detections.sort(key=lambda x: x['confidence'], reverse=True)

    return img_final, detections


@app.route('/')
def index():
    return send_file('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    """Main detection route: receives image, selects golden PCB (optional), analyses defects, returns results."""

    try:
        data = request.get_json()

        # Decode test PCB image
        test_data = data['test'].split(',')[1]
        test_bytes = base64.b64decode(test_data)
        test_arr = np.frombuffer(test_bytes, np.uint8)
        test_img = cv2.imdecode(test_arr, cv2.IMREAD_COLOR)

        if test_img is None:
            return jsonify({'success': False, 'error': 'Failed to decode test image'}), 400

        use_auto_detect = data.get('autoDetect', False)

        matched_golden_name = None
        match_score = 0
        golden_img = None

        if use_auto_detect:
            # Auto-select golden PCB (based on ORB feature matching)
            if not golden_pcbs:
                return jsonify({
                    'success': False,
                    'error': 'No golden PCB images found in golden/ folder.'
                }), 400

            test_img_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

            golden_img_gray, matched_golden_name, match_score = identify_golden_pcb(test_img_gray, golden_pcbs)

            if golden_img_gray is None or match_score == 0:
                return jsonify({
                    'success': False,
                    'error': 'Could not match test PCB with any golden images.'
                }), 400

            # Convert grayscale golden → BGR for subtraction
            golden_img = cv2.cvtColor(golden_img_gray, cv2.COLOR_GRAY2BGR)

        else:
            # Manual golden image upload path
            if 'golden' not in data:
                return jsonify({
                    'success': False,
                    'error': 'Golden PCB image is required unless autoDetect is enabled.'
                }), 400

            golden_data = data['golden'].split(',')[1]
            golden_bytes = base64.b64decode(golden_data)
            golden_arr = np.frombuffer(golden_bytes, np.uint8)
            golden_img = cv2.imdecode(golden_arr, cv2.IMREAD_COLOR)

            if golden_img is None:
                return jsonify({'success': False, 'error': 'Failed to decode golden image'}), 400

        # Perform defect analysis
        img_final, detections = process_images(golden_img, test_img)

        # Encode annotated image into base64
        _, buffer = cv2.imencode('.jpg', img_final)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Prepare CSV export
        csv_data = "Defect_ID,Defect_Type,Confidence_%,BBox_X1,BBox_Y1,BBox_X2,BBox_Y2,Center_X,Center_Y\n"
        for idx, det in enumerate(detections, 1):
            csv_data += f"{idx},{det['class']},{det['confidence']},{det['bbox'][0]},{det['bbox'][1]},{det['bbox'][2]},{det['bbox'][3]},{det['center_x']},{det['center_y']}\n"

        response_data = {
            'success': True,
            'image': f'data:image/jpeg;base64,{img_base64}',
            'detections': detections,
            'total': len(detections),
            'csv': csv_data
        }

        # Include auto-detection info if used
        if use_auto_detect and matched_golden_name:
            response_data['matchedGolden'] = matched_golden_name
            response_data['matchScore'] = match_score

        return jsonify(response_data)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("Loading model...")
    load_model()

    print("Model loaded successfully.")
    print("Loading golden PCB images...")
    load_golden_pcbs()

    print("Starting server...")
    app.run(debug=True, port=5000)
