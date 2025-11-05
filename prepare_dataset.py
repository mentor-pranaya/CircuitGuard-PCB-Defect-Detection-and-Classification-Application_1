import os
import shutil

source_rois = 'defect_rois'
output_dir = 'defect_rois_organized'

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(source_rois):
    defect_class = None
    parts = filename.split('_')
    if len(parts) >= 3:
        defect_class = parts[1]
        # standardize capitalization (Spurious_copper)
        if defect_class.lower() == 'spurious':
            # handle files like 'spurious_copper' (multi underscore)
            defect_class = parts[1] + '_' + parts[2]
        defect_class = defect_class.replace('spurious_copper', 'Spurious_copper').replace('mouse_bite', 'Mouse_bite').replace('spur', 'Spur').replace('short', 'Short').replace('open_circuit', 'Open_circuit').replace('missing_hole', 'Missing_hole')
    if defect_class is None:
        print(f"Cannot find class in: {filename}")
        continue
    class_dir = os.path.join(output_dir, defect_class)
    os.makedirs(class_dir, exist_ok=True)
    src_path = os.path.join(source_rois, filename)
    dst_path = os.path.join(class_dir, filename)
    shutil.copy2(src_path, dst_path)
    print(f"Copied {filename} to {defect_class}")

print("\nOrganization done: ROI images are sorted by class in defect_rois_organized/")
