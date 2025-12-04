#This Script identifies golden pcb from test pcb using ORB Matching
import cv2
import numpy as np
import os

# Load all golden PCB images from a folder
def load_golden_pcbs(folder_path):
    golden_pcbs = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                golden_pcbs.append((filename, img))
    return golden_pcbs


# Identify which golden PCB matches the test PCB
def identify_golden_pcb(test_img_gray, golden_pcbs, nfeatures=2000):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    kp_test, des_test = orb.detectAndCompute(test_img_gray, None)
    if des_test is None:
        return None, 0  # cannot match without descriptors

    best_name = None
    best_matches = 0

    for name, golden_img in golden_pcbs:
        kp_golden, des_golden = orb.detectAndCompute(golden_img, None)
        if des_golden is None:
            continue

        matches = bf.match(des_test, des_golden)
        matches = sorted(matches, key=lambda x: x.distance)

        # number of matches indicates similarity
        score = len(matches)

        if score > best_matches:
            best_matches = score
            best_name = name

    return best_name, best_matches

# Example usage
if __name__ == "__main__":
    GOLDEN_FOLDER = "golden/"       # folder with 12 golden PCB images
    TEST_IMAGE_PATH = "test/10.jpg"     # the PCB image to identify

    golden_pcbs = load_golden_pcbs(GOLDEN_FOLDER)

    test_img = cv2.imread(TEST_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

    golden_name, score = identify_golden_pcb(test_img, golden_pcbs)

    if golden_name:
        print("Best match:", golden_name)
        print("Matching score:", score)
    else:
        print("Could not identify a matching golden PCB.")
