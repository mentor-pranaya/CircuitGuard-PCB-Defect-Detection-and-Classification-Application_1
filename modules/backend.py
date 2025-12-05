import cv2
import numpy as np
import torch
import timm
from torchvision import transforms
from PIL import Image
import os
import glob
import subprocess

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "/content/drive/MyDrive/CircuitGuard_Project/models/circuitguard_efficientnet_b4_v2.pth"
DRIVE_ZIP = "/content/drive/MyDrive/CircuitGuard_Project/new PCB _ds.zip"
LOCAL_TEMPLATE_DIR = "/content/new PCB _ds/PCB_USED"
IMG_SIZE = 128

class CircuitGuardBackend:
    def __init__(self):
        self.model = None
        self.classes = None
        self.templates = []
        self.ensure_data_exists()
        self.load_model()
        self.load_templates()

    def ensure_data_exists(self):
        if not os.path.exists(LOCAL_TEMPLATE_DIR):
            if os.path.exists(DRIVE_ZIP):
                subprocess.run(["unzip", "-q", DRIVE_ZIP, "-d", "/content/"])

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            return
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            self.classes = checkpoint['classes']
            self.model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=len(self.classes))
            self.model.load_state_dict(checkpoint['model'])
            self.model.to(DEVICE)
            self.model.eval()
        except Exception:
            pass

    def load_templates(self):
        files = glob.glob(os.path.join(LOCAL_TEMPLATE_DIR, "*.JPG")) + \
                glob.glob(os.path.join(LOCAL_TEMPLATE_DIR, "*.jpg"))
        for f in files:
            img = cv2.imread(f)
            if img is not None:
                self.templates.append({
                    "name": os.path.basename(f),
                    "img": img,
                    "gray": cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                })

    def find_best_template(self, img_test_cv):
        if not self.templates: return None
        gray_test = cv2.cvtColor(img_test_cv, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=1000)
        kp_test, des_test = orb.detectAndCompute(gray_test, None)
        if des_test is None: return None
        
        best_match = None
        max_matches = 0
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        for temp in self.templates:
            kp_temp, des_temp = orb.detectAndCompute(temp['gray'], None)
            if des_temp is None: continue
            matches = matcher.match(des_test, des_temp)
            if len(matches) > max_matches:
                max_matches = len(matches)
                best_match = temp['img']
        return best_match

    def align_images(self, img_template, img_test):
        gray_temp = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
        gray_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=5000)
        kp1, des1 = orb.detectAndCompute(gray_temp, None)
        kp2, des2 = orb.detectAndCompute(gray_test, None)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good = matches[:int(len(matches) * 0.15)]
        if len(good) < 4: return None
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
        h, w = gray_temp.shape
        return cv2.warpPerspective(img_test, H, (w, h))

    def predict_roi(self, roi_img):
        roi_pil = Image.fromarray(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
        tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        t = tf(roi_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = self.model(t)
            prob = torch.softmax(out, dim=1)
            conf, pred = torch.max(prob, 1)
        return self.classes[pred.item()], conf.item()

    def run_pipeline(self, pil_image):
        img_test = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        img_template = self.find_best_template(img_test)
        if img_template is None:
            return None, None, "Could not find a matching reference board."

        aligned = self.align_images(img_template, img_test)
        if aligned is None: return None, None, "Alignment Failed"

        blur_t = cv2.GaussianBlur(img_template, (5, 5), 0)
        blur_a = cv2.GaussianBlur(aligned, (5, 5), 0)
        diff = cv2.absdiff(blur_t, blur_a)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        viz_img = aligned.copy()
        results = []
        
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) < 50: continue
            x, y, w, h = cv2.boundingRect(cnt)
            pad = 60
            H, W = aligned.shape[:2]
            y1, y2 = max(0, y-pad), min(H, y+h+pad)
            x1, x2 = max(0, x-pad), min(W, x+w+pad)
            crop = aligned[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            label, conf = self.predict_roi(crop)
            
            cv2.rectangle(viz_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(viz_img, f"{label} {conf:.0%}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            results.append({"ID": i+1, "Defect": label, "Confidence": f"{conf:.1%}"})
            
        return viz_img, results, "Success"
