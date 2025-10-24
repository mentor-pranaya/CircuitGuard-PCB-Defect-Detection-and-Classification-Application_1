import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

#PATH SETUP 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_ROOT = os.path.join(BASE_DIR, "..", "template")
TEST_DIR = os.path.join(BASE_DIR, "..", "test")
OUTPUT_ROOT = os.path.join(BASE_DIR, "..", "output")
ROI_ROOT = os.path.join(OUTPUT_ROOT, "rois")

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(ROI_ROOT, exist_ok=True)

#  MODULE 1: IMAGE PREPROCESSING & SUBTRACTION 

def align_images(template, image):
    """Optional feature-based alignment using ORB."""
    orb = cv2.ORB_create(2000)
    k1, d1 = orb.detectAndCompute(template, None)
    k2, d2 = orb.detectAndCompute(image, None)
    if d1 is None or d2 is None:
        return image
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(d1, d2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) > 10:
        src_pts = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        h, w = template.shape
        aligned = cv2.warpPerspective(image, M, (w, h))
        return aligned
    return image


def process_pair(template_path, test_path, out_mask_path, kernel_size=(3,3)):
    """Compute subtraction mask between template and test images."""
    templ = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    test = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    if templ is None or test is None:
        raise ValueError(f"Error reading {template_path} or {test_path}")

    test = cv2.resize(test, (templ.shape[1], templ.shape[0]))
    test = align_images(templ, test)

    # Gaussian blur
    templ_b = cv2.GaussianBlur(templ, (5,5), 0)
    test_b  = cv2.GaussianBlur(test, (5,5), 0)

    # Absolute difference
    diff = cv2.absdiff(templ_b, test_b)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    diff_cl = clahe.apply(diff)

    # Otsu thresholding
    _, th = cv2.threshold(diff_cl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    mask = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    cv2.imwrite(out_mask_path, mask)
    return templ, test, diff, mask


#  MODULE 2: CONTOUR DETECTION & ROI EXTRACTION

def extract_rois_from_mask(mask, orig_image, min_area=80, pad=8):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # pad and clip
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(orig_image.shape[1], x + w + pad)
        y2 = min(orig_image.shape[0], y + h + pad)
        roi = orig_image[y1:y2, x1:x2].copy()
        rois.append(((x1, y1, x2, y2), roi, area))
    return rois


def run_pipeline():
    """Runs Module 1 + 2 together."""
    categories = [d for d in os.listdir(TEMPLATE_ROOT) if os.path.isdir(os.path.join(TEMPLATE_ROOT, d))]
    csv_path = os.path.join(ROI_ROOT, "rois_metadata.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "mask_path", "test_image", "roi_name", "x1", "y1", "x2", "y2", "area"])

        for cat in categories:
            print(f"\n=== Processing category: {cat} ===")
            t_dir = os.path.join(TEMPLATE_ROOT, cat)
            test_imgs = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg','.png','.jpeg'))])
            t_imgs = sorted([f for f in os.listdir(t_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])

            out_cat = os.path.join(OUTPUT_ROOT, cat)
            os.makedirs(out_cat, exist_ok=True)
            roi_cat = os.path.join(ROI_ROOT, cat)
            os.makedirs(roi_cat, exist_ok=True)

            for i, tname in enumerate(t_imgs):
                if i >= len(test_imgs):
                    break
                tpath = os.path.join(t_dir, tname)
                testpath = os.path.join(TEST_DIR, test_imgs[i])
                mask_path = os.path.join(out_cat, f"mask_{i+1:03d}.png")

                templ, test, diff, mask = process_pair(tpath, testpath, mask_path)
                print(f"  -> Saved mask: {mask_path}")

                # ROI extraction
                rois = extract_rois_from_mask(mask, test, min_area=100, pad=10)
                for j, (bbox, roi_img, area) in enumerate(rois):
                    roi_name = f"{os.path.splitext(tname)[0]}_roi_{j+1:02d}.png"
                    roi_path = os.path.join(roi_cat, roi_name)
                    cv2.imwrite(roi_path, roi_img)
                    x1, y1, x2, y2 = bbox
                    writer.writerow([cat, mask_path, test_imgs[i], roi_name, x1, y1, x2, y2, area])

    print("\n✅ Pipeline complete.")
    print(f"ROI metadata saved at: {csv_path}")


if __name__ == "__main__":
    run_pipeline()
