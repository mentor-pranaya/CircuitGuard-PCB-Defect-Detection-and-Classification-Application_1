import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import torch
from torchvision import transforms
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model classes and transform (adjust classes as per your model)
CLASS_NAMES = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
TARGET_SIZE = (128, 128)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def load_model(model_path, num_classes=6):
    from torchvision import models
    import torch.nn as nn

    model = models.efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def find_best_template(defect_img, template_dir):
    defect_gray = cv2.cvtColor(defect_img, cv2.COLOR_BGR2GRAY)
    h, w = defect_gray.shape
    best_score = -1
    best_template = None
    best_template_path = None

    for fname in os.listdir(template_dir):
        fpath = os.path.join(template_dir, fname)
        template = cv2.imread(fpath)
        if template is None:
            continue
        template_resized = cv2.resize(template, (w, h))
        template_gray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY)
        score = ssim(defect_gray, template_gray)
        if score > best_score:
            best_score = score
            best_template = template_resized
            best_template_path = fpath
    return best_template, best_template_path

def align_images(template_gray, defect_gray):
    orb = cv2.ORB_create(nfeatures=5000)
    kp1, des1 = orb.detectAndCompute(template_gray, None)
    kp2, des2 = orb.detectAndCompute(defect_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:50]

    points1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
    height, width = template_gray.shape
    aligned_defect = cv2.warpPerspective(defect_gray, H, (width, height))
    return aligned_defect, H

def get_diff_mask(template_gray, aligned_defect):
    diff = cv2.absdiff(template_gray, aligned_defect)
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    morph = cv2.erode(thresh, kernel, iterations=1)
    morph = cv2.dilate(morph, kernel, iterations=1)
    return diff, morph

def predict_roi(model, roi_img):
    input_tensor = transform(roi_img)
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        _, pred = torch.max(probs, 1)
        class_idx = pred.item()
        confidence = probs[0, class_idx].item()
    return CLASS_NAMES[class_idx], confidence

def extract_rois_and_annotate(model, defect_img_color, defect_mask):
    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated_img = defect_img_color.copy()
    rois = []
    defect_log = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 40:  # filter noise
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        padding = 12
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(defect_img_color.shape[1], x + w + padding)
        y2 = min(defect_img_color.shape[0], y + h + padding)

        roi = defect_img_color[y1:y2, x1:x2]

        label, conf = predict_roi(model, roi)

        # Draw bounding box + label
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(annotated_img, f"{label} ({conf*100:.1f}%)", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        rois.append((label, roi))
        defect_log.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "label": label, "confidence": conf})

    return annotated_img, rois, defect_log

def process_uploaded_defect_image(defect_img, template_dir, model):
    # Find best template image
    template_img, template_path = find_best_template(defect_img, template_dir)
    if template_img is None:
        raise FileNotFoundError("No defect-free template images found")

    # Convert to grayscale for alignment/subtraction
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    defect_gray = cv2.cvtColor(defect_img, cv2.COLOR_BGR2GRAY)

    # Align defect to template
    aligned_defect_gray, H = align_images(template_gray, defect_gray)

    # Difference and mask
    diff_img, defect_mask = get_diff_mask(template_gray, aligned_defect_gray)

    # Extract ROIs and annotate
    annotated_img, rois, defect_log = extract_rois_and_annotate(model, defect_img, defect_mask)

    return {
        "template_img": template_img,
        "aligned_defect_gray": aligned_defect_gray,
        "diff_img": diff_img,
        "defect_mask": defect_mask,
        "annotated_img": annotated_img,
        "rois": rois,
        "defect_log": defect_log,
        "template_path": template_path
    }
