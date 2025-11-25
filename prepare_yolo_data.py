import os
import glob
import shutil
import random
import xml.etree.ElementTree as ET
# from tqdm import tqdm

# Configuration
BASE_DIR = r"c:\Users\jayza\OneDrive\Desktop\Infosys\PCB_DATASET\PCB_DATASET"
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "Annotations")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_DIR = r"c:\Users\jayza\OneDrive\Desktop\Infosys\yolo_dataset"

CLASSES = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']

def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def prepare_data():
    # Create directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

    print("Converting annotations...")
    
    all_files = []
    
    # Collect all files
    for class_name in CLASSES:
        class_dir = os.path.join(ANNOTATIONS_DIR, class_name)
        xml_files = glob.glob(os.path.join(class_dir, "*.xml"))
        
        for xml_file in xml_files:
            all_files.append((class_name, xml_file))

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_files)
    split_idx = int(len(all_files) * 0.8)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    def process_files(files, split):
        print(f"Processing {split} ({len(files)} files)...")
        for i, (class_name, xml_file) in enumerate(files):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(files)}")
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Image info
            filename = root.find('filename').text
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            # Find corresponding image
            img_path = os.path.join(IMAGES_DIR, class_name, filename)
            if not os.path.exists(img_path):
                print(f"MISSING: {img_path}")
                continue

            # Convert labels
            label_data = []
            for obj in root.iter('object'):
                cls = obj.find('name').text
                # Normalize to match our CLASSES list (which is Capitalized)
                # Map lowercase xml name to Capitalized folder name
                if cls.lower() == 'missing_hole': cls_mapped = 'Missing_hole'
                elif cls.lower() == 'mouse_bite': cls_mapped = 'Mouse_bite'
                elif cls.lower() == 'open_circuit': cls_mapped = 'Open_circuit'
                elif cls.lower() == 'short': cls_mapped = 'Short'
                elif cls.lower() == 'spur': cls_mapped = 'Spur'
                elif cls.lower() == 'spurious_copper': cls_mapped = 'Spurious_copper'
                else: continue
                
                cls_id = CLASSES.index(cls_mapped)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert_bbox((w, h), b)
                label_data.append(f"{cls_id} {bb[0]} {bb[1]} {bb[2]} {bb[3]}")

            # Save
            if label_data:
                # Copy image
                shutil.copy(img_path, os.path.join(OUTPUT_DIR, 'images', split, filename))
                
                # Save label
                label_path = os.path.join(OUTPUT_DIR, 'labels', split, filename.replace('.jpg', '.txt'))
                with open(label_path, 'w') as f:
                    f.write('\n'.join(label_data))
            else:
                print(f"SKIPPED (No Labels): {filename}")

    process_files(train_files, 'train')
    process_files(val_files, 'val')

    # Create data.yaml
    yaml_content = f"""
path: {OUTPUT_DIR}
train: images/train
val: images/val

nc: {len(CLASSES)}
names: {CLASSES}
    """
    with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        f.write(yaml_content.strip())

    print(f"Data preparation complete! Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    with open("log.txt", "w") as log:
        try:
            log.write("Starting...\n")
            prepare_data()
            log.write("Done!\n")
        except Exception as e:
            log.write(f"Error: {e}\n")
