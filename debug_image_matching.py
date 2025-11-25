import os
import glob
import xml.etree.ElementTree as ET

BASE_DIR = r"c:\Users\jayza\OneDrive\Desktop\Infosys\PCB_DATASET\PCB_DATASET"
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "Annotations")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

print("Debugging Image Matching...")

# Get first XML file
xml_files = glob.glob(os.path.join(ANNOTATIONS_DIR, "*", "*.xml"))
if not xml_files:
    print("No XML files found!")
    exit()

sample_xml = xml_files[0]
print(f"Sample XML: {sample_xml}")

tree = ET.parse(sample_xml)
root = tree.getroot()
filename = root.find('filename').text
print(f"Filename in XML: {filename}")

# Check where it expects the image
class_name = os.path.basename(os.path.dirname(sample_xml))
expected_path = os.path.join(IMAGES_DIR, class_name, filename)
print(f"Expected Image Path: {expected_path}")

if os.path.exists(expected_path):
    print("Image FOUND!")
else:
    print("Image NOT FOUND!")
    # List files in that directory to see what's there
    dir_to_check = os.path.dirname(expected_path)
    if os.path.exists(dir_to_check):
        print(f"Files in {dir_to_check}:")
        print(os.listdir(dir_to_check)[:5])
    else:
        print(f"Directory {dir_to_check} does not exist")
