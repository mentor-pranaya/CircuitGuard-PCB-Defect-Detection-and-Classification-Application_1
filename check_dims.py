import cv2
import os

template_path = r"PCB_DATASET/PCB_DATASET/PCB_USED/01.JPG"
test_path = r"PCB_DATASET/PCB_DATASET/images/Missing_hole/01_missing_hole_01.jpg"

if os.path.exists(template_path) and os.path.exists(test_path):
    img1 = cv2.imread(template_path)
    img2 = cv2.imread(test_path)
    
    print(f"Template: {template_path}, Shape: {img1.shape}")
    print(f"Test: {test_path}, Shape: {img2.shape}")
else:
    print("Files not found")
