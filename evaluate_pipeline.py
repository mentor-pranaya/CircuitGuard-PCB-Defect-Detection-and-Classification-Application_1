import os
import cv2
import torch
import timm
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
import preprocessing

# Configuration
MODEL_PATH = "best_model.pth"
DATASET_ROOT = Path("PCB_DATASET/PCB_DATASET")
IMAGES_DIR = DATASET_ROOT / "images"
TEMPLATE_DIR = DATASET_ROOT / "PCB_USED"
OUTPUT_DIR = Path("evaluation_results")
NUM_CLASSES = 6
CLASS_NAMES = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
print(f"Loading model from {MODEL_PATH}...")
model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_template_path(filename):
    prefix = filename.split('_')[0]
    return TEMPLATE_DIR / f"{prefix}.JPG"

def evaluate():
    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Select 1 image from each class for testing
    test_images = []
    for defect_dir in IMAGES_DIR.iterdir():
        if defect_dir.is_dir():
            images = list(defect_dir.glob("*.jpg"))
            if images:
                # Pick the last one as a "test" image
                test_images.append(images[-1])

    print(f"Selected {len(test_images)} images for evaluation...")

    for img_path in test_images:
        print(f"Processing {img_path.name}...")
        template_path = get_template_path(img_path.name)
        
        if not template_path.exists():
            print("Template not found.")
            continue

        # 1. Load Images
        img_test = cv2.imread(str(img_path))
        img_template = cv2.imread(str(template_path))

        # 2. Pipeline: Align -> Subtract -> ROI
        try:
            img_aligned = preprocessing.align_images(img_test, img_template)
            mask = preprocessing.subtract_images(img_aligned, img_template)
            rois = preprocessing.extract_rois(img_aligned, mask, min_area=50)
        except Exception as e:
            print(f"Pipeline failed: {e}")
            continue

        # 3. Classify ROIs
        img_result = img_aligned.copy()
        detections = []

        for (x, y, w, h) in rois:
            # Pad crop
            pad = 10
            h_img, w_img = img_aligned.shape[:2]
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad)
            y2 = min(h_img, y + h + pad)
            
            crop = img_aligned[y1:y2, x1:x2]
            
            # Convert to PIL
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            input_tensor = transform(crop_pil).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            class_id = predicted.item()
            label = CLASS_NAMES[class_id]
            conf_score = confidence.item()

            if conf_score > 0.5:
                detections.append((label, conf_score))
                # Draw box
                color = (0, 255, 0) # Green
                cv2.rectangle(img_result, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img_result, f"{label} {conf_score:.2f}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save result
        save_path = OUTPUT_DIR / f"eval_{img_path.name}"
        cv2.imwrite(str(save_path), img_result)
        print(f"Saved result to {save_path}. Detections: {detections}")

    print("Evaluation complete. Check 'evaluation_results' folder.")

if __name__ == "__main__":
    evaluate()
