import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import timm
from PIL import Image
from typing import List, Tuple, Dict
import glob
import math

# CONFIGURATION CONSTANTS 
TEMPLATE_ROOT = 'C:\\Users\\Lenovo\\PCB Python\\PCB_USED' 
MODEL_CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'efficientnet_b4_pcb_classifier_best.pth')
MODEL_NAME = 'efficientnet_b4'
IMAGE_SIZE = 128
MIN_DEFECT_AREA = 100 

# Standard normalization parameters (matching training.py)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spurious_copper', 'spur'] 
NUM_CLASSES = len(CLASS_NAMES)

# Initialize ORB for feature matching (used for both identification and alignment)
ORB = cv2.ORB_create(500)

# --- 1. NEW TEMPLATE IDENTIFICATION STEP ---

def find_best_template_match(test_image_path: str, template_root: str) -> Tuple[str, str]:

    test_image_gray = cv2.imread(test_image_path, 0)
    if test_image_gray is None:
        raise FileNotFoundError(f"Test image not found at {test_image_path}")

    # Detect and compute features for the test image once
    kp_test, des_test = ORB.detectAndCompute(test_image_gray, None)
    
    if des_test is None or len(des_test) < 10:
        raise ValueError("Could not find enough features on the test image for matching.")

    best_match = {'path': None, 'id': None, 'good_matches': 0, 'confidence': 0}
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Iterate over all templates in the root directory
    template_files = glob.glob(os.path.join(template_root, '*.jpg')) # Assumes JPG
    
    if not template_files:
        raise FileNotFoundError(f"No template images found in TEMPLATE_ROOT: {template_root}")

    for template_path in template_files:
        template_id = os.path.splitext(os.path.basename(template_path))[0]
        template_gray = cv2.imread(template_path, 0)
        
        if template_gray is None:
            print(f"Skipping unreadable template: {template_path}")
            continue

        kp_temp, des_temp = ORB.detectAndCompute(template_gray, None)

        if des_temp is None or len(des_temp) < 10:
            continue
        
        # Match features
        matches = matcher.match(des_temp, des_test)
        
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:int(len(matches) * 0.15)] # Take top 15% as good
        
        num_good_matches = len(good_matches)
        
        if num_good_matches > best_match['good_matches']:
            best_match['good_matches'] = num_good_matches
            best_match['path'] = template_path
            best_match['id'] = template_id

    if best_match['path'] is None or best_match['good_matches'] < 10:
        raise Exception("Could not confidently identify the corresponding golden PCB.")
        
    print(f"Automatically identified Golden PCB: {best_match['id']} with {best_match['good_matches']} good matches.")
    return best_match['path'], best_match['id']

def load_inference_model(num_classes: int, model_name: str, checkpoint_path: str):
    """Loads the EfficientNet model structure and trained weights."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(model_name, pretrained=False) 
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please check the path.")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval() 
    
    print(f"Model {model_name} loaded successfully onto {device} from {os.path.basename(checkpoint_path)}.")
    return model, device

def predict_single_roi(model, device, roi_image: np.ndarray, class_names: List[str]) -> Tuple[str, float]:
    """Preprocesses a single ROI image (numpy array) and runs classification."""
    pil_img = Image.fromarray(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ])
    
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)

    probabilities = torch.softmax(output, dim=1)
    confidence, predicted_index = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted_index.item()]
    confidence_score = confidence.item() * 100
    
    return predicted_class, confidence_score


# --- 3. IMAGE PROCESSING UTILITIES (Slightly adjusted alignment input) ---

def align_and_subtract(template_path: str, test_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aligns the test image to the template, performs subtraction, and thresholds."""
    
    # Load Template (Color and Grayscale)
    template_color = cv2.imread(template_path)
    template_gray = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)

    if template_color is None:
        raise FileNotFoundError(f"Template image not found at {template_path}")

    # Convert test image to grayscale 
    test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)


    kp1, des1 = ORB.detectAndCompute(template_gray, None)
    kp2, des2 = ORB.detectAndCompute(test_gray, None)
    
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
         raise ValueError("Could not find enough features for robust alignment.")
    
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des1, des2), key=lambda x: x.distance)
    good_matches = matches[:int(len(matches) * 0.10)]
    
    points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    h, _ = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
    if h is None:
        raise ValueError("Homography calculation failed. Alignment not possible.")

    # Warp both color and grayscale test images
    height, width = template_gray.shape
    aligned_test_gray = cv2.warpPerspective(test_gray, h, (width, height))
    aligned_test_color = cv2.warpPerspective(test_image, h, (width, height))
    
    # --- Subtraction & Thresholding (Module 1 Logic) ---
    difference_map = cv2.absdiff(template_gray, aligned_test_gray)
    _, defect_mask = cv2.threshold(difference_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    
    return aligned_test_color, defect_mask, template_color

def extract_defect_contours(defect_mask: np.ndarray, aligned_image_color: np.ndarray, min_area: int) -> List[Dict]:
    """Performs morphological ops, contour detection, and ROI extraction (Module 2 Logic)."""
    
    kernel = np.ones((5, 5), np.uint8) 
    mask_open = cv2.erode(defect_mask, kernel, iterations=1)
    mask_open = cv2.dilate(mask_open, kernel, iterations=1)

    dilated_mask = cv2.dilate(mask_open, kernel, iterations=1)
    cleaned_mask = cv2.erode(dilated_mask, kernel, iterations=1)
    
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    defect_rois = []
    
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        roi_patch = aligned_image_color[y:y+h, x:x+w]
        
        defect_rois.append({
            'roi_patch': roi_patch, 
            'bbox': (x, y, w, h)
        })
        
    return defect_rois

def draw_annotations(image: np.ndarray, results: List[Dict]) -> np.ndarray:
    """Draws bounding boxes and labels onto the image (Annotation Step)."""
    annotated_image = image.copy()
    BOX_COLOR = (0, 0, 255)
    TEXT_COLOR = (255, 255, 255) 
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 1.0    
    THICKNESS = 3       
    for res in results:
        x, y, w, h = res['bbox']
        label = res['prediction']
        conf = res['confidence']

        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), BOX_COLOR, THICKNESS)
        
        text = f"{label} ({conf:.1f}%)"
        (text_width, text_height), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
        
        cv2.rectangle(annotated_image, (x, y - text_height - baseline), (x + text_width, y), BOX_COLOR, -1)
        cv2.putText(annotated_image, text, (x, y - baseline), FONT, FONT_SCALE, TEXT_COLOR, THICKNESS-1, cv2.LINE_AA)

    return annotated_image


# --- 4. THE MAIN PIPELINE FUNCTION (Modified) ---

def full_inference_pipeline(test_image_path: str) -> Tuple[np.ndarray, List[Dict], str]:
    
    # 1. Setup
    model, device = load_inference_model(NUM_CLASSES, MODEL_NAME, MODEL_CHECKPOINT_PATH)
    test_image_color = cv2.imread(test_image_path)

    if test_image_color is None:
        raise FileNotFoundError(f"Test image not found at {test_image_path}")

    template_path, template_id = find_best_template_match(test_image_path, TEMPLATE_ROOT)
    
    # 3. Image Processing (Alignment, Subtraction, Contours)
    aligned_test_color, defect_mask, _ = align_and_subtract(template_path, test_image_color)
    defect_rois = extract_defect_contours(defect_mask, aligned_test_color, MIN_DEFECT_AREA)
    
    print(f"Found {len(defect_rois)} potential defect region(s).")

    # 4. Classification
    final_results = []
    
    if not defect_rois:
        print("No defects found. Returning clean image.")
        return aligned_test_color, final_results, template_id 

    for defect in defect_rois:
        predicted_class, confidence = predict_single_roi(model, device, defect['roi_patch'], CLASS_NAMES)
        
        final_results.append({
            'bbox': defect['bbox'],
            'prediction': predicted_class,
            'confidence': confidence
        })
        print(f"  Defect at {defect['bbox']} classified as: {predicted_class} ({confidence:.2f}%)")

    # 5. Annotation
    annotated_image = draw_annotations(aligned_test_color, final_results)
    
    print("Inference pipeline complete. Annotated image generated.")
    return annotated_image, final_results, template_id


if __name__ == "__main__":
    # Example Usage (replace with actual path)
    SAMPLE_TEST_IMAGE = 'C:\\Users\\Lenovo\\PCB Python\\images\\Short\\01\\01_short_01.jpg' 
    
    if SAMPLE_TEST_IMAGE.startswith('C:\\Users\\Lenovo\\PCB Python\\images'):
        print("\n!!! ACTION REQUIRED: UPDATE SAMPLE_TEST_IMAGE and TEMPLATE_ROOT PATHS !!!")
    else:
        try:
            output_image, results, template_id = full_inference_pipeline(SAMPLE_TEST_IMAGE)
            output_path = os.path.join(os.path.dirname(SAMPLE_TEST_IMAGE), f'annotated_output_id_{template_id}.jpg')
            cv2.imwrite(output_path, output_image)
            print(f"\nFinal Annotated Image saved to: {output_path}")

        except Exception as e:
            print(f"\nFATAL ERROR in pipeline execution: {e}")