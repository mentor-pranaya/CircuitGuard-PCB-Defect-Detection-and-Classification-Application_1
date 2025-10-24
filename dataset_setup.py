import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===========================================
#  MODULE 1: PROCESS ALL IMAGE PAIRS
# ===========================================

base_dir = os.path.dirname(os.path.abspath(__file__))  # points to scripts/
template_root = os.path.join(base_dir, "..", "template")  # go up 1 level to AI_PCB/template
test_dir = os.path.join(base_dir, "..", "test")          # go up 1 level to AI_PCB/test
output_root = os.path.join(base_dir, "..", "output")     # go up 1 level to AI_PCB/output


# ✅ Ensure directories exist
if not os.path.exists(template_root):
    raise FileNotFoundError(f"❌ Template folder not found: {template_root}")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"❌ Test folder not found: {test_dir}")
os.makedirs(output_root, exist_ok=True)

# ✅ Collect defect folders
defect_folders = [f for f in os.listdir(template_root) if os.path.isdir(os.path.join(template_root, f))]
if not defect_folders:
    raise ValueError("⚠️ No defect subfolders found inside template!")

print(f"✅ Found defect types: {defect_folders}")
print("🔄 Starting batch processing...")

# Loop through each defect category
for defect in defect_folders:
    defect_path = os.path.join(template_root, defect)
    output_dir = os.path.join(output_root, defect)
    os.makedirs(output_dir, exist_ok=True)

    template_images = sorted([
        f for f in os.listdir(defect_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    test_images = sorted([
        f for f in os.listdir(test_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not template_images or not test_images:
        print(f"⚠️ Skipping {defect}: missing images.")
        continue

    # Process each template image (paired by index)
    for idx, temp_name in enumerate(template_images):
        if idx >= len(test_images):
            break

        temp_path = os.path.join(defect_path, temp_name)
        test_path = os.path.join(test_dir, test_images[idx])

        template = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
        test = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

        if template is None or test is None:
            print(f"⚠️ Error reading pair {temp_name} / {test_images[idx]}")
            continue

        # Resize test to match template
        test = cv2.resize(test, (template.shape[1], template.shape[0]))

        # Gaussian blur
        template_blur = cv2.GaussianBlur(template, (5, 5), 0)
        test_blur = cv2.GaussianBlur(test, (5, 5), 0)

        # Subtraction
        diff = cv2.absdiff(template_blur, test_blur)

        # Otsu Thresholding
        _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological filtering
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Save output
        output_name = f"mask_{idx+1:03d}.png"
        output_path = os.path.join(output_dir, output_name)
        cv2.imwrite(output_path, mask)

        print(f"✅ Saved: {output_path}")

print("\n🎯 All image pairs processed successfully!")
print(f"🗂️ Output masks stored in: {output_root}")

# ===========================================
# Optional: Show 1 sample from each defect type
# ===========================================
for defect in defect_folders:
    sample_dir = os.path.join(output_root, defect)
    masks = sorted([f for f in os.listdir(sample_dir) if f.lower().endswith('.png')])
    template_images = sorted([f for f in os.listdir(os.path.join(template_root, defect)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    test_images = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    if not masks or not template_images or not test_images:
        continue

    mask_path = os.path.join(sample_dir, masks[0])
    temp_path = os.path.join(template_root, defect, template_images[0])
    test_path = os.path.join(test_dir, test_images[0])

    template = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
    test = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    plt.figure(figsize=(10, 4))
    plt.suptitle(f"Defect Type: {defect}", fontsize=14)
    plt.subplot(1, 3, 1); plt.imshow(template, cmap='gray'); plt.title('Template'); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(test, cmap='gray'); plt.title('Test'); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(mask, cmap='gray'); plt.title('Defect Mask'); plt.axis('off')
    plt.tight_layout()
    plt.show()
