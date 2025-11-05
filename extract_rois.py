import os
import cv2
import xml.etree.ElementTree as ET

# Paths
annotations_folder = '.'   # XML annotation files are in the project root directory
images_folders = ['dataset/test_images', 'images']  # Image folders
output_folder = 'defect_rois'  # Output folder for cropped ROI images

os.makedirs(output_folder, exist_ok=True)

# Function to find image file in the image folders
def find_image(filename):
    for folder in images_folders:
        img_path = os.path.join(folder, filename)
        if os.path.exists(img_path):
            return img_path
    return None

# Function to extract ROIs from one XML annotation file
def extract_rois_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    img_filename = root.find('filename').text
    img_filepath = find_image(img_filename)
    if img_filepath is None:
        print(f'Image not found: {img_filename}')
        return
    img = cv2.imread(img_filepath)
    if img is None:
        print(f'Could not load image: {img_filepath}')
        return

    for idx, obj in enumerate(root.findall('object')):
        cls = obj.find('name').text
        bbox = obj.find('bndbox')
        x1 = int(float(bbox.find('xmin').text))
        y1 = int(float(bbox.find('ymin').text))
        x2 = int(float(bbox.find('xmax').text))
        y2 = int(float(bbox.find('ymax').text))
        roi = img[y1:y2, x1:x2]
        outname = f"{os.path.splitext(img_filename)[0]}_{cls}_roi_{idx}.jpg"
        cv2.imwrite(os.path.join(output_folder, outname), roi)
        print(f"Saved: {outname}")

# Main loop over all XML files in the annotations folder
for xml_fname in os.listdir(annotations_folder):
    if not xml_fname.endswith('.xml'):
        continue
    xml_file = os.path.join(annotations_folder, xml_fname)
    extract_rois_from_xml(xml_file)

print("\nExtraction complete: all ROIs for all classes are saved in defect_rois/")
