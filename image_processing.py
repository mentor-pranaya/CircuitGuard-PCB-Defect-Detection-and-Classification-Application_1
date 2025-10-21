import cv2
import numpy as np
import os

template_path = 'dataset/defect_free'
test_path = 'dataset/test_images'
output_path = 'output_detected'

# Load one defect-free image (template)
template_files = [f for f in os.listdir(template_path) if os.path.isfile(os.path.join(template_path, f))]
if not template_files:
    print("No defect-free images found in dataset/defect_free!")
    exit()

template_file = os.path.join(template_path, template_files[0])
template = cv2.imread(template_file, cv2.IMREAD_GRAYSCALE)
if template is None:
    print("Error loading the template image!")
    exit()

# Get all test image files (ignore folders)
test_files = [f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))]
if not test_files:
    print("No test images found in dataset/test_images!")
    exit()

for test_file_name in test_files:
    test_file = os.path.join(test_path, test_file_name)
    test_img = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)
    if test_img is None:
        print(f"Error loading test image: {test_file_name}")
        continue

    # Resize for consistency
    if template.shape != test_img.shape:
        test_img = cv2.resize(test_img, (template.shape[1], template.shape[0]))

    # Step 1: Image subtraction
    diff = cv2.absdiff(template, test_img)

    # Step 2: Thresholding to highlight differences
    _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)

    # Step 3: Morph operations to clean noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    # Step 4: Contour detection
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Save the output image with debug print
    out_file = os.path.join(output_path, f'detected_{test_file_name}')
    success = cv2.imwrite(out_file, output)
    print(f"Saved? {success} to {out_file}")

print("All test images processed and saved.")
