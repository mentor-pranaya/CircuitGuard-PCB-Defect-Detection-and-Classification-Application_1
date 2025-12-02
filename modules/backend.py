import os
import cv2
import numpy as np
import torch
import timm
from torchvision import transforms
from PIL import Image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "/content/drive/MyDrive/CircuitGuard_Project/models/circuitguard_efficientnet_b4_v2.pth"
IMG_SIZE = 128

class CircuitGuardBackend:
    def __init__(self):
        self.model = None
        self.classes = []
        self.load_model()

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model not found at {MODEL_PATH}")
            self.model = None
            self.classes = []
            return
        
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            self.classes = checkpoint.get('classes', [])
            if not self.classes:
                raise ValueError("Classes not found in checkpoint.")
            
            self.model = timm.create_model(
                'efficientnet_b4',
                pretrained=False,
                num_classes=len(self.classes)
            )
            self.model.load_state_dict(checkpoint['model'])
            self.model.to(DEVICE)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None
            self.classes = []

    def align_images(self, img_template, img_test):
        gray_temp = cv2.cvtColor(img_template, cv2.COLOR_RGB2GRAY)
        gray_test = cv2.cvtColor(img_test, cv2.COLOR_RGB2GRAY)
        
        orb = cv2.ORB_create(nfeatures=5000)
        kp1, des1 = orb.detectAndCompute(gray_temp, None)
        kp2, des2 = orb.detectAndCompute(gray_test, None)

        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            return None
        
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        if len(matches) == 0:
            return None
        
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:int(len(matches) * 0.15)]
        if len(good_matches) < 4:
            return None
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None
        
        h, w = gray_temp.shape
        aligned = cv2.warpPerspective(img_test, H, (w, h))
        return aligned

    def get_defect_mask(self, img_template, img_aligned):
        blur_t = cv2.GaussianBlur(img_template, (5, 5), 0)
        blur_a = cv2.GaussianBlur(img_aligned, (5, 5), 0)
        diff = cv2.absdiff(blur_t, blur_a)
        gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        return mask

    def predict_roi(self, roi_img):
        if self.model is None or not self.classes:
            raise RuntimeError("Model not loaded.")
        
        roi_pil = Image.fromarray(roi_img)
        tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
        tensor = tf(roi_pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            out = self.model(tensor)
            prob = torch.softmax(out, dim=1)
            conf, pred = torch.max(prob, 1)
        
        label = self.classes[pred.item()]
        return label, conf.item()

    def run_pipeline(self, pil_temp, pil_test):
        if self.model is None or not self.classes:
            return None, None, "Model not loaded"
        
        img_t = np.array(pil_temp.convert("RGB"))
        img_test = np.array(pil_test.convert("RGB"))
        
        aligned = self.align_images(img_t, img_test)
        if aligned is None:
            return None, None, "Alignment Failed"
        
        mask = self.get_defect_mask(img_t, aligned)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        viz_img = aligned.copy()
        results = []
        
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) < 50:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            pad = 60
            H_img, W_img = aligned.shape[:2]
            y1, y2 = max(0, y - pad), min(H_img, y + h + pad)
            x1, x2 = max(0, x - pad), min(W_img, x + w + pad)
            
            crop = aligned[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            try:
                label, conf = self.predict_roi(crop)
            except Exception as e:
                print(f"Prediction error on ROI {i+1}: {e}")
                continue
            
            cv2.rectangle(viz_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            label_text = f"{label} ({conf:.0%})"
            cv2.putText(
                viz_img,
                label_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            
            results.append({
                "ID": i + 1,
                "Defect Type": label,
                "Confidence": f"{conf:.2%}"
            })
        
        return viz_img, results, "Success"
