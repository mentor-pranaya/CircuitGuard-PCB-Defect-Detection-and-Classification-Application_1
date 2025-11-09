#Spliting The Dataset for Training, Testing and Validation
import os
import shutil
import random

# Path to your ROI folder
SOURCE_DIR = r"C:/Users/Badal/Desktop/SpringBoard/Output_data/ROI"
OUTPUT_DIR = r"C:/Users/Badal/Desktop/SpringBoard/Model_Data/ROI"

# Allowed image types
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

# Train/Val/Test split ratio
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15  # test = 0.15

def create_dirs(base, classes):
    for split in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(base, split, cls), exist_ok=True)

def get_images_recursive(folder):
    """Return all image file paths recursively inside folder."""
    image_paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                image_paths.append(os.path.join(root, f))
    return image_paths

def split_dataset():
    classes = [d for d in os.listdir(SOURCE_DIR)
               if os.path.isdir(os.path.join(SOURCE_DIR, d))]

    create_dirs(OUTPUT_DIR, classes)

    for cls in classes:
        class_path = os.path.join(SOURCE_DIR, cls)

        # Recursively collect all .jpg files inside subfolders
        images = get_images_recursive(class_path)

        print(f"\nClass '{cls}' → Found {len(images)} images")

        if len(images) == 0:
            print("⚠ No images found here! Check the folder structure.")
            continue

        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * TRAIN_SPLIT)
        n_val = int(n_total * VAL_SPLIT)

        train_files = images[:n_train]
        val_files = images[n_train:n_train+n_val]
        test_files = images[n_train+n_val:]

        # Copy images to destination folders
        for img_path in train_files:
            shutil.copy(img_path, os.path.join(OUTPUT_DIR, "train", cls,
                                               os.path.basename(img_path)))

        for img_path in val_files:
            shutil.copy(img_path, os.path.join(OUTPUT_DIR, "val", cls,
                                               os.path.basename(img_path)))

        for img_path in test_files:
            shutil.copy(img_path, os.path.join(OUTPUT_DIR, "test", cls,
                                               os.path.basename(img_path)))

        print(f"✓ Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    print("\nDataset split completed successfully!")
    print(f"Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    split_dataset()
