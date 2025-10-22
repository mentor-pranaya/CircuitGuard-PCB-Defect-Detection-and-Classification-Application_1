import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


def find_all_test_folders(base_test_folder):
    """Find all subfolders in the test directory"""
    test_folders = []
    for item in os.listdir(base_test_folder):
        item_path = os.path.join(base_test_folder, item)
        if os.path.isdir(item_path):
            test_folders.append(item_path)
            print(f"📁 Found test folder: {item}")
    return test_folders


def find_image_pairs(template_folder, test_folder):
    """Find matching template and test image pairs between folders"""
    template_files = [f for f in os.listdir(
        template_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    test_files = [f for f in os.listdir(test_folder) if f.lower().endswith(
        ('.png', '.jpg', '.jpeg', '.bmp'))]

    print(f"   Template images: {len(template_files)}")
    print(f"   Test images: {len(test_files)}")

    # Try to find pairs by matching filenames
    pairs = []

    for tpl_file in template_files:
        tpl_base = Path(tpl_file).stem.lower()

        # Look for test files with similar names
        for test_file in test_files:
            test_base = Path(test_file).stem.lower()

            # Different matching strategies
            if (tpl_base in test_base or
                test_base in tpl_base or
                tpl_base.replace('template', '') == test_base.replace('test', '') or
                    tpl_base.replace('good', '') == test_base.replace('defect', '')):

                pairs.append((tpl_file, test_file))
                break

    # If no pairs found by name, try sequential matching
    if not pairs and len(template_files) == len(test_files):
        print("   🔄 Using sequential matching...")
        for i in range(min(len(template_files), len(test_files))):
            pairs.append((template_files[i], test_files[i]))

    print(f"   🔗 Found {len(pairs)} image pairs")
    return pairs


def process_single_pair(template_path, test_path, output_dir, pair_name):
    """Process one template-test image pair"""
    # Load images
    template = cv2.imread(template_path)
    test = cv2.imread(test_path)

    if template is None or test is None:
        print(f"❌ Could not load images: {template_path} or {test_path}")
        return None

    # Preprocess - resize and convert to grayscale
    if template.shape != test.shape:
        test = cv2.resize(test, (template.shape[1], template.shape[0]))

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    # Image subtraction
    difference = cv2.absdiff(template_gray, test_gray)

    # Otsu thresholding
    _, binary_mask = cv2.threshold(
        difference, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleaning
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    # Create result with defects highlighted in red
    result_overlay = test.copy()
    colored_mask = np.zeros_like(test)
    colored_mask[cleaned_mask == 255] = [0, 0, 255]  # Red color
    result_overlay = cv2.addWeighted(result_overlay, 0.7, colored_mask, 0.3, 0)

    # Save results
    cv2.imwrite(os.path.join(
        output_dir, f"{pair_name}_difference.jpg"), difference)
    cv2.imwrite(os.path.join(
        output_dir, f"{pair_name}_mask.jpg"), cleaned_mask)
    cv2.imwrite(os.path.join(
        output_dir, f"{pair_name}_result.jpg"), result_overlay)

    return cleaned_mask


def main():
    print("🚀 Starting Batch Image Preprocessing & Subtraction")
    print("=" * 50)

    # PROVIDE YOUR PATHS HERE:
    template_folder = r"C:\Users\jayza\OneDrive\Desktop\Infosys\PCB_DATASET\PCB_DATASET\PCB_USED"
    base_test_folder = r"C:\Users\jayza\OneDrive\Desktop\Infosys\PCB_DATASET\PCB_DATASET\images"

    # Create main output directory
    main_output_dir = "results/module1_output"
    os.makedirs(main_output_dir, exist_ok=True)

    # Check if folders exist
    if not os.path.exists(template_folder):
        print(f"❌ Template folder not found: {template_folder}")
        return
    if not os.path.exists(base_test_folder):
        print(f"❌ Test base folder not found: {base_test_folder}")
        return

    # Find all test folders
    test_folders = find_all_test_folders(base_test_folder)

    if not test_folders:
        print("❌ No test folders found!")
        return

    total_pairs_processed = 0

    # Process each test folder
    for test_folder in test_folders:
        folder_name = os.path.basename(test_folder)
        print(f"\n🎯 Processing defect type: {folder_name}")

        # Create subfolder for this defect type
        output_dir = os.path.join(main_output_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)

        # Find image pairs between template folder and this test folder
        pairs = find_image_pairs(template_folder, test_folder)

        if not pairs:
            print(f"   ⚠️  No image pairs found for {folder_name}")
            continue

        # Process all pairs in this folder
        folder_pairs_count = 0
        for i, (tpl_file, test_file) in enumerate(pairs, 1):
            template_path = os.path.join(template_folder, tpl_file)
            test_path = os.path.join(test_folder, test_file)

            pair_name = f"{folder_name}_{Path(tpl_file).stem}_vs_{Path(test_file).stem}"
            print(f"   [{i}/{len(pairs)}] Processing: {tpl_file} vs {test_file}")

            result = process_single_pair(
                template_path, test_path, output_dir, pair_name)
            if result is not None:
                folder_pairs_count += 1

        total_pairs_processed += folder_pairs_count
        print(f"   ✅ Completed {folder_pairs_count} pairs for {folder_name}")

    print(f"\n🎉 BATCH PROCESSING COMPLETED!")
    print("=" * 50)
    print(f"📊 Summary:")
    print(f"   • Template folder: {template_folder}")
    print(f"   • Test folders processed: {len(test_folders)}")
    print(f"   • Total image pairs processed: {total_pairs_processed}")
    print(f"   • Results saved in: {main_output_dir}")
    print(f"   • Each defect type has its own subfolder")


if __name__ == "__main__":
    main()
