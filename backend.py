import io
import cv2
import numpy as np
import torch
import timm
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import preprocessing

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = "best_model.pth"
YOLO_PATH = "best_yolo.pt"
NUM_CLASSES = 6
CLASS_NAMES = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Models ---

# 1. EfficientNet (Reference Mode)
print(f"Loading EfficientNet from {MODEL_PATH}...")
try:
    model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("EfficientNet loaded.")
except Exception as e:
    print(f"Error loading EfficientNet: {e}")

# 2. YOLOv8 (AI Mode)
print(f"Loading YOLO from {YOLO_PATH}...")
try:
    yolo_model = YOLO(YOLO_PATH)
    print("YOLO loaded.")
except Exception as e:
    print(f"Error loading YOLO: {e}")
    yolo_model = None

# --- Helper Functions ---

def read_imagefile(file) -> np.ndarray:
    image = Image.open(io.BytesIO(file)).convert('RGB')
    return np.array(image)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Endpoints ---

@app.post("/predict")
async def predict(
    test: UploadFile = File(...), 
    template: UploadFile = File(None), 
    mode: str = Form("reference")
):
    # Read Test Image
    img_test_bytes = await test.read()
    img_test = read_imagefile(img_test_bytes)
    img_test_cv = cv2.cvtColor(img_test, cv2.COLOR_RGB2BGR)
    
    results = []

    if mode == "ai":
        # --- AI MODE (YOLO) ---
        if yolo_model is None:
            return {"error": "YOLO model not loaded."}
        
        # Run Inference
        yolo_results = yolo_model(img_test_cv)
        
        for r in yolo_results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = CLASS_NAMES[cls]
                
                if conf > 0.25: # Confidence threshold
                    results.append({
                        "bbox": [int(x1), int(y1), int(x2-x1), int(y2-y1)], # x, y, w, h
                        "label": label,
                        "confidence": conf
                    })

    else:
        # --- REFERENCE MODE (Original) ---
        if not template:
            return {"error": "Template image required for Reference Mode"}
            
        img_template = read_imagefile(await template.read())
        img_template_cv = cv2.cvtColor(img_template, cv2.COLOR_RGB2BGR)

        try:
            img_aligned = preprocessing.align_images(img_test_cv, img_template_cv)
            mask = preprocessing.subtract_images(img_aligned, img_template_cv)
            rois = preprocessing.extract_rois(img_aligned, mask, min_area=100)
        except Exception as e:
            return {"error": f"Preprocessing failed: {str(e)}"}

        # Classify ROIs
        for (x, y, w, h) in rois:
            # Pad crop
            pad = 10
            h_img, w_img = img_aligned.shape[:2]
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad)
            y2 = min(h_img, y + h + pad)
            
            crop = img_aligned[y1:y2, x1:x2]
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            input_tensor = transform(crop_pil).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf_score, pred_class = torch.max(probs, 1)
                label = CLASS_NAMES[pred_class.item()]

            if conf_score > 0.5:
                # Add padding for visualization
                vis_pad = 20
                vx1 = max(0, x - vis_pad)
                vy1 = max(0, y - vis_pad)
                vx2 = min(w_img, x + w + vis_pad)
                vy2 = min(h_img, y + h + vis_pad)

                results.append({
                    "bbox": [vx1, vy1, vx2 - vx1, vy2 - vy1], 
                    "label": label,
                    "confidence": float(conf_score)
                })

    return {"defects": results}

@app.get("/")
def read_root():
    return {"status": "CircuitGuard Backend Running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
