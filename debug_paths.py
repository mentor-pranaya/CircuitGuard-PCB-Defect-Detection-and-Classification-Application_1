import os
import glob

BASE_DIR = r"c:\Users\jayza\OneDrive\Desktop\Infosys\PCB_DATASET\PCB_DATASET"
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "Annotations")

print(f"Checking {ANNOTATIONS_DIR}")
if os.path.exists(ANNOTATIONS_DIR):
    print("Directory exists")
    files = glob.glob(os.path.join(ANNOTATIONS_DIR, "*", "*.xml"))
    print(f"Found {len(files)} XML files")
else:
    print("Directory does NOT exist")
