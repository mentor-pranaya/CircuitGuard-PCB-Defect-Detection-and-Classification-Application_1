import cv2
import numpy as np
import os

BASE_DIR = r"C:\Users\LENOVO\OneDrive\Desktop\AI_PCB"
TEMPLATE_DIR = os.path.join(BASE_DIR, "template")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_results")
ROI_DIR = os.path.join(OUTPUT_DIR, "defect_rois")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ROI_DIR, exist_ok=True)

def contour_detection(input_img, category, name):
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defect_visual = input_img.copy()
    count = 0

    category_dir = os.path.join(ROI_DIR, category)
    os.makedirs(category_dir, exist_ok=True)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(defect_visual, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(defect_visual, category, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            roi = input_img[y:y + h, x:x + w]
            roi_path = os.path.join(category_dir, f"{name}_defect_{count}.png")
            cv2.imwrite(roi_path, roi)
            count += 1

    save_path = os.path.join(OUTPUT_DIR, f"{category}_{name}_defects.png")
    cv2.imwrite(save_path, defect_visual)
    print(f"Saved {count} defect ROIs for {category} → {name}")

for category in os.listdir(TEMPLATE_DIR):
    category_path = os.path.join(TEMPLATE_DIR, category)
    if os.path.isdir(category_path):
        for file in os.listdir(category_path):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(category_path, file)
                name = os.path.splitext(file)[0]
                print(f"Processing: {category} → {file}")
                img = cv2.imread(img_path)
                if img is not None:
                    contour_detection(img, category, name)
                else:
                    print(f"Could not read image: {file}")
