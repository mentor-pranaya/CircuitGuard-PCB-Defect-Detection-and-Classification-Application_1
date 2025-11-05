import os
import cv2
import numpy as np
import json
import csv

class ROIExtractor:
    def __init__(self, min_area=100, max_area=10000):
        self.min_area = min_area
        self.max_area = max_area
        self.kernel = np.ones((3, 3), np.uint8)
    
    def preprocess_mask(self, mask):
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.kernel)
        return cleaned
    
    def detect_contours(self, mask):
        cleaned_mask = self.preprocess_mask(mask)
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [
            c for c in contours if self.min_area < cv2.contourArea(c) < self.max_area
        ]
        return filtered_contours, cleaned_mask
    
    def extract_bounding_boxes(self, contours):
        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append({'x': x, 'y': y, 'width': w, 'height': h})
        return boxes
    
    def crop_and_resize(self, image, boxes, padding=10):
        crops = []
        max_w, max_h = 0, 0

        for bbox in boxes:
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            x1, y1 = max(0, x - padding), max(0, y - padding)
            x2, y2 = min(image.shape[1], x + w + padding), min(image.shape[0], y + h + padding)
            crop = image[y1:y2, x1:x2]

            if crop.size > 0:
                max_w = max(max_w, crop.shape[1])
                max_h = max(max_h, crop.shape[0])
                crops.append({
                    'image': crop,
                    'bbox': (x1, y1, x2 - x1, y2 - y1)
                })

        for crop in crops:
            crop['image'] = cv2.resize(crop['image'], (max_w, max_h))

        return crops
    
    def visualize_results(self, image, contours, boxes, output_path):
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
        for b in boxes:
            x, y, w, h = b['x'], b['y'], b['width'], b['height']
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite(output_path, vis)

def process_roi_extraction():
    extractor = ROIExtractor(min_area=50, max_area=5000)
    input_dir = 'Result'
    output_dir = 'ROI_Results'
    
    os.makedirs(output_dir, exist_ok=True)
    cropped_dir = os.path.join(output_dir, 'cropped_defects')
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(cropped_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    defect_types = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
    total_defects, results_summary = 0, {}

    label_file = os.path.join(output_dir, 'labels.csv')
    with open(label_file, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'defect_type', 'x', 'y', 'width', 'height']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for defect_type in defect_types:
            print("Processing", defect_type)
            mask_dir = os.path.join(input_dir, defect_type)
            if not os.path.isdir(mask_dir):
                print("Folder not found:", mask_dir)
                continue

            mask_files = [f for f in os.listdir(mask_dir) if f.endswith('_mask.png')]
            type_defects = 0

            for mask_file in mask_files:
                mask_path = os.path.join(mask_dir, mask_file)
                mask = cv2.imread(mask_path, 0)
                if mask is None:
                    print("Skipped:", mask_path)
                    continue

                base_name = mask_file.replace('_mask.png', '')
                aligned_path = os.path.join(mask_dir, f"{base_name}_aligned.jpg")
                original = cv2.imread(aligned_path, 0)
                if original is None:
                    print("Missing aligned image:", aligned_path)
                    continue

                contours, _ = extractor.detect_contours(mask)
                boxes = extractor.extract_bounding_boxes(contours)
                crops = extractor.crop_and_resize(original, boxes)

                for i, crop in enumerate(crops):
                    defect_filename = f"{defect_type}_{base_name}_defect_{i+1}.jpg"
                    save_path = os.path.join(cropped_dir, defect_filename)
                    cv2.imwrite(save_path, crop['image'])
                    (x, y, w, h) = crop['bbox']
                    writer.writerow({'filename': defect_filename, 'defect_type': defect_type, 'x': x, 'y': y, 'width': w, 'height': h})
                    type_defects += 1

                viz_path = os.path.join(vis_dir, f"{defect_type}_{base_name}_viz.jpg")
                extractor.visualize_results(original, contours, boxes, viz_path)

            results_summary[defect_type] = type_defects
            total_defects += type_defects

    summary = {
        'total_defects_extracted': total_defects,
        'defects_by_type': results_summary,
        'extraction_parameters': {
            'min_area': extractor.min_area,
            'max_area': extractor.max_area
        }
    }

    with open(os.path.join(output_dir, 'extraction_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("Extraction complete")
    print("Total defects extracted:", total_defects)
    print("Labels saved to:", label_file)
    return summary

if __name__ == "__main__":
    process_roi_extraction()
