import cv2
import numpy as np
import os
from pathlib import Path


def detect_contours_and_extract_rois(mask_path, original_image_path, output_dir, defect_type):
    """Main function to detect contours and extract ROIs"""
    print(f"🔍 Processing: {Path(mask_path).name}")

    # Load the binary mask from Module 1
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    original = cv2.imread(original_image_path)

    if mask is None or original is None:
        print(f"❌ Could not load images: {mask_path} or {original_image_path}")
        return []

    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"   📊 Found {len(contours)} potential defects")

    roi_info = []
    valid_defects = 0

    # Process each contour
    for i, contour in enumerate(contours):
        # Filter out very small contours (noise)
        area = cv2.contourArea(contour)
        if area < 50:
            continue

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Add padding
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(original.shape[1] - x, w + 2 * padding)
        h = min(original.shape[0] - y, h + 2 * padding)

        # Crop defect region
        defect_roi = original[y:y+h, x:x+w]

        if defect_roi.size == 0:
            continue

        # Save cropped defect
        roi_filename = f"{defect_type}_defect_{valid_defects+1}.jpg"
        roi_path = os.path.join(output_dir, roi_filename)
        cv2.imwrite(roi_path, defect_roi)

        roi_info.append({
            'file': roi_filename,
            'position': (x, y),
            'size': (w, h),
            'area': area,
            'contour': contour
        })

        valid_defects += 1

    print(f"   ✅ Extracted {valid_defects} valid defects")
    return roi_info


def main():
    print("🚀 Starting Module 2: Contour Detection & ROI Extraction")

    module1_results_dir = "results/module1_output"
    module2_output_dir = "results/module2_output"
    os.makedirs(module2_output_dir, exist_ok=True)

    total_defects_found = 0

    # Process each defect type folder
    for defect_type in os.listdir(module1_results_dir):
        defect_path = os.path.join(module1_results_dir, defect_type)

        if not os.path.isdir(defect_path):
            continue

        print(f"\n🎯 Processing: {defect_type}")

        # Create output folders
        roi_output_dir = os.path.join(
            module2_output_dir, defect_type, "cropped_defects")
        os.makedirs(roi_output_dir, exist_ok=True)

        # Process each mask file
        mask_files = [f for f in os.listdir(
            defect_path) if f.endswith('_mask.jpg')]

        for mask_file in mask_files:
            mask_path = os.path.join(defect_path, mask_file)
            base_name = mask_file.replace('_mask.jpg', '')
            original_image_path = os.path.join(
                defect_path, f"{base_name}_result.jpg")

            if not os.path.exists(original_image_path):
                continue

            # Detect contours and extract ROIs
            roi_info = detect_contours_and_extract_rois(
                mask_path, original_image_path, roi_output_dir, defect_type
            )
            total_defects_found += len(roi_info)

    print(f"\n🎉 MODULE 2 COMPLETED!")
    print(f"📊 Total defects extracted: {total_defects_found}")


if __name__ == "__main__":
    main()
