import os
from collections import defaultdict

def detect_defect_files(dataset_path):
    defect_files = defaultdict(list)

    if not os.path.exists(dataset_path):
        print(f"Error: Folder '{dataset_path}' not found!")
        return

    filenames = os.listdir(dataset_path)

    for fname in filenames:
        file_path = os.path.join(dataset_path, fname)

        # Skip non-files, hidden files, or non-images
        if not os.path.isfile(file_path) or fname.startswith('.') or not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        parts = fname.split('_')
        if len(parts) > 1:
            defect = parts[1]
            defect_files[defect].append(fname)
        else:
            print(f"⚠️ Skipping file (unexpected format): {fname}")

    if defect_files:
        print("\nDetected defect types and corresponding files:")
        for defect, files in defect_files.items():
            print(f"\n🧩 {defect} ({len(files)} files):")
            for f in files:
                print(f"  - {f}")
    else:
        print("No valid defect types detected.")


if __name__ == "__main__":
    dataset_path = "dataset/test_images"
    detect_defect_files(dataset_path)
