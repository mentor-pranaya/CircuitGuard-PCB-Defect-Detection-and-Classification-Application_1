import cv2
import numpy as np
import os

def align_images(img_test, img_template):
    # Convert images to grayscale
    gray_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create(5000)

    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(gray_test, None)
    kp2, des2 = orb.detectAndCompute(gray_template, None)

    # Match features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(des1, des2, None)

    # Sort matches by score
    matches = sorted(matches, key=lambda x: x.distance, reverse=False)

    # Keep top 15% matches
    numGoodMatches = int(len(matches) * 0.15)
    matches = matches[:numGoodMatches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography to warp image
    height, width, channels = img_template.shape
    img_aligned = cv2.warpPerspective(img_test, h, (width, height))

    return img_aligned

def subtract_images(img_aligned, img_template):
    # Convert to grayscale
    gray_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference
    diff = cv2.absdiff(gray_aligned, gray_template)

    # Apply Gaussian blur to reduce noise
    diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)

    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(diff_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh_clean = cv2.morphologyEx(thresh_clean, cv2.MORPH_CLOSE, kernel)

    return thresh_clean

def main():
    template_path = r"PCB_DATASET/PCB_DATASET/PCB_USED/01.JPG"
    test_path = r"PCB_DATASET/PCB_DATASET/images/Missing_hole/01_missing_hole_01.jpg"
    output_path = "subtraction_result.jpg"

    if not os.path.exists(template_path) or not os.path.exists(test_path):
        print("Images not found")
        return

    print("Loading images...")
    img_template = cv2.imread(template_path)
    img_test = cv2.imread(test_path)

    print("Aligning images...")
    img_aligned = align_images(img_test, img_template)

    print("Subtracting images...")
    defect_mask = subtract_images(img_aligned, img_template)

    # Find contours on the mask
    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes on the aligned image
    img_result = img_aligned.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) > 100: # Filter small noise
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_result, (x, y), (x+w, y+h), (0, 0, 255), 3)

    cv2.imwrite(output_path, img_result)
    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    main()
