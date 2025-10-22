import os
import cv2
import matplotlib.pyplot as plt


def check_image_structure():
    base_path = "PCD_DATASET/images"
    defect_folders = os.listdir(base_path)

    print("🔍 Checking image structure in defect folders...")

    for defect_type in defect_folders[:2]:  # Check first 2 defect types
        defect_path = os.path.join(base_path, defect_type)
        if os.path.isdir(defect_path):
            files = os.listdir(defect_path)
            image_files = [f for f in files if f.lower().endswith(
                ('.png', '.jpg', '.jpeg', '.bmp'))]

            print(f"\n📁 {defect_type}:")
            print(f"   Images found: {len(image_files)}")

            if image_files:
                # Show first 3 image names to understand naming pattern
                for img_file in image_files[:3]:
                    print(f"   - {img_file}")

                # Load and display properties of first image
                first_image_path = os.path.join(defect_path, image_files[0])
                img = cv2.imread(first_image_path)
                if img is not None:
                    print(f"   📏 Image shape: {img.shape}")
                    print(f"   🎨 Data type: {img.dtype}")


check_image_structure()
