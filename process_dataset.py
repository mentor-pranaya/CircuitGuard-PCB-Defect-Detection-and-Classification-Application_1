import os
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path

DATASET_ROOT = Path("PCB_DATASET/PCB_DATASET")
IMAGES_DIR = DATASET_ROOT / "images"
ANNOTATIONS_DIR = DATASET_ROOT / "Annotations"
OUTPUT_DIR = Path("processed_data")

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        objects.append({'name': name, 'bbox': (xmin, ymin, xmax, ymax)})
    return objects

def main():
    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Iterate over defect types in Annotations directory
    for defect_dir in ANNOTATIONS_DIR.iterdir():
        if not defect_dir.is_dir():
            continue
            
        defect_type = defect_dir.name
        print(f"Processing {defect_type}...")
        
        output_subdir = OUTPUT_DIR / defect_type
        output_subdir.mkdir(exist_ok=True)
        
        for xml_path in defect_dir.glob("*.xml"):
            # Find corresponding image
            # XML: 01_missing_hole_01.xml -> Image: 01_missing_hole_01.jpg
            img_name = xml_path.stem + ".jpg"
            img_path = IMAGES_DIR / defect_type / img_name
            
            if not img_path.exists():
                # Try .JPG extension just in case
                img_name = xml_path.stem + ".JPG"
                img_path = IMAGES_DIR / defect_type / img_name
                
            if not img_path.exists():
                print(f"Image not found for {xml_path.name}")
                continue
                
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Error reading image: {img_path}")
                continue

            # Parse XML
            objects = parse_xml(xml_path)
            
            for i, obj in enumerate(objects):
                xmin, ymin, xmax, ymax = obj['bbox']
                
                # Add padding (optional, but good for context)
                pad = 10
                height, width = img.shape[:2]
                xmin = max(0, xmin - pad)
                ymin = max(0, ymin - pad)
                xmax = min(width, xmax + pad)
                ymax = min(height, ymax + pad)
                
                crop = img[ymin:ymax, xmin:xmax]
                
                if crop.size == 0:
                    print(f"Empty crop for {xml_path.name} object {i}")
                    continue

                save_name = f"{xml_path.stem}_crop_{i}.jpg"
                cv2.imwrite(str(output_subdir / save_name), crop)

    print("Processing complete. ROIs extracted from annotations.")

if __name__ == "__main__":
    main()
