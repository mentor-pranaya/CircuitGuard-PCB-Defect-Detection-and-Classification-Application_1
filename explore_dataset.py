import os
import cv2


def explore_dataset():
    base_path = "C:/Users/jayza/OneDrive/Desktop/Infosys/PCB_DATASET/PCB_DATASET"

    print("🔍 Exploring Dataset Structure...")

    # Check if main folder exists
    if not os.path.exists(base_path):
        print("❌ PCD_DATASET folder not found!")
        return

    # List all items in main folder
    print("\n📁 Main folder contents:")
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            print(f"📂 Folder: {item}")
        else:
            print(f"📄 File: {item}")

    # Specifically check images folder
    images_path = os.path.join(base_path, "images")
    if os.path.exists(images_path):
        print(f"\n📸 Contents of 'images' folder:")
        image_files = os.listdir(images_path)
        for img_file in image_files[:10]:
            print(f"   - {img_file}")
        print(
            f"   ... and {len(image_files) - 10} more files" if len(image_files) > 10 else "")
    else:
        print("❌ 'images' folder not found!")


if __name__ == "__main__":
    explore_dataset()
