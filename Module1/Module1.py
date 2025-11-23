import cv2
import numpy as np
import os
import glob

# CONFIG
BASE_DIR = "."
INPUT_DIR = os.path.join(BASE_DIR, "images")

# Locations where templates might exist
TEMPLATE_DIRS = [
    os.path.join(BASE_DIR, "PCB_USED"),
    os.path.join(BASE_DIR, "templates"),
    BASE_DIR
]

# Output folders
MASK_DIR = os.path.join(BASE_DIR, "PROCESSED_MASKS")
CROP_DIR = os.path.join(BASE_DIR, "DEFECT_CROPS")
VISUAL_DIR = os.path.join(BASE_DIR, "ANNOTATED_RESULTS")

for d in [MASK_DIR, CROP_DIR, VISUAL_DIR]:
    os.makedirs(d, exist_ok=True)


# TEMPLATE SEARCH
def find_template(board_id):
    """
    Attempts to locate a template image that matches the test board ID.
    Tries several filename patterns and searches known template folders.
    """
    possible_names = [
        f"{board_id}.jpg",
        f"{board_id}_temp.jpg",
        f"{board_id}_template.jpg",
        f"temp_{board_id}.jpg"
    ]

    # Direct search in main template directories
    for folder in TEMPLATE_DIRS:
        if not os.path.exists(folder):
            continue
        for name in possible_names:
            path = os.path.join(folder, name)
            if os.path.exists(path):
                return path

    # Recursive search if not found
    for folder in TEMPLATE_DIRS:
        if not os.path.exists(folder):
            continue
        for root, _, files in os.walk(folder):
            for f in files:
                if f in possible_names:
                    return os.path.join(root, f)

    return None


# IMAGE ALIGNMENT
def align_images(img_test, img_template):
    """
    Aligns the test PCB image to the template using ORB keypoint matching.
    """
    gray_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
    gray_temp = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=5000)
    kp1, des1 = orb.detectAndCompute(gray_test, None)
    kp2, des2 = orb.detectAndCompute(gray_temp, None)

    # If features are missing, skip alignment
    if des1 is None or des2 is None:
        return img_test

    # Match ORB features
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Keep top 15% good matches
    num_good = int(len(matches) * 0.15)
    matches = matches[:num_good]

    # Extract matched keypoints
    pts_test = np.zeros((len(matches), 2), dtype=np.float32)
    pts_temp = np.zeros((len(matches), 2), dtype=np.float32)

    for i, m in enumerate(matches):
        pts_test[i] = kp1[m.queryIdx].pt
        pts_temp[i] = kp2[m.trainIdx].pt

    # Compute homography and warp alignment
    h_matrix, _ = cv2.findHomography(pts_test, pts_temp, cv2.RANSAC)
    h, w, _ = img_template.shape
    return cv2.warpPerspective(img_test, h_matrix, (w, h))


# PROCESSING A TEST–TEMPLATE PAIR
def process_pair(test_path, template_path, filename):
    """
    Runs alignment, subtraction, thresholding, contour extraction,
    and saves mask + defect crops + annotated output.
    """
    img_test = cv2.imread(test_path)
    img_temp = cv2.imread(template_path)

    if img_test is None or img_temp is None:
        return 0

    # Attempt alignment
    try:
        aligned_test = align_images(img_test, img_temp)
    except:
        aligned_test = img_test

    # Subtraction for defect detection
    diff = cv2.absdiff(img_temp, aligned_test)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Binary mask using Otsu thresholding
    _, thresh = cv2.threshold(diff_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(os.path.join(MASK_DIR, f"mask_{filename}"), mask)

    # Extract defect contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated_img = aligned_test.copy()
    defect_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:
            defect_count += 1

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Save cropped defect region
            pad = 10
            roi = aligned_test[
                max(0, y - pad): y + h + pad,
                max(0, x - pad): x + w + pad
            ]

            crop_name = f"{os.path.splitext(filename)[0]}_defect{defect_count}.jpg"
            cv2.imwrite(os.path.join(CROP_DIR, crop_name), roi)

    cv2.imwrite(os.path.join(VISUAL_DIR, f"annotated_{filename}"), annotated_img)
    return defect_count


# MAIN PIPELINE
def main():
    # Collect test PCB images
    all_test_images = glob.glob(os.path.join(INPUT_DIR, "**", "*.jpg"), recursive=True)

    # Exclude templates if mixed in the folder
    all_test_images = [f for f in all_test_images if "temp" not in os.path.basename(f).lower()]

    print(f"Found {len(all_test_images)} test images.")

    success_count = 0
    missing_templates = set()

    for img_path in all_test_images:
        filename = os.path.basename(img_path)

        # Extract board ID from filename
        try:
            board_id = filename.split("_")[0]
        except:
            continue

        template_path = find_template(board_id)

        if template_path:
            try:
                count = process_pair(img_path, template_path, filename)
                print(f"{filename}: {count} defects")
                success_count += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            if board_id not in missing_templates:
                print(f"No template found for board ID: {board_id}")
                missing_templates.add(board_id)

    print(f"Processing completed. {success_count} images processed.")
    print(f"Annotated results stored in: {VISUAL_DIR}")


if __name__ == "__main__":
    main()
