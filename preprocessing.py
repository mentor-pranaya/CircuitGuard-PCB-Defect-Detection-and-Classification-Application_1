import cv2
import numpy as np

def align_images(img_test, img_template):
    """
    Aligns img_test to match img_template using ORB feature matching and Homography.
    """
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

    if len(matches) < 4:
        print("Not enough matches found for alignment.")
        return img_test

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
    """
    Computes difference between aligned image and template, applies thresholding and morphology.
    Returns a binary mask of defects.
    """
    # Convert to grayscale
    gray_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference
    diff = cv2.absdiff(gray_aligned, gray_template)

    # Apply Gaussian blur to reduce noise
    diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)

    # Apply fixed thresholding (better for controlled lighting)
    _, thresh = cv2.threshold(diff_blur, 25, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh_clean = cv2.morphologyEx(thresh_clean, cv2.MORPH_CLOSE, kernel)
    
    return thresh_clean

def extract_rois(img, mask, min_area=100):
    """
    Finds contours in the mask and extracts bounding boxes.
    Returns a list of (x, y, w, h) tuples.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            rois.append((x, y, w, h))
    return rois
