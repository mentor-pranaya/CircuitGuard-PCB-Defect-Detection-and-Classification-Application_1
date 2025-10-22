import os
import cv2
import numpy as np
import sys
TEMPLATE_ROOT = 'C:\\Users\\Lenovo\\PCB Python\\PCB_USED' 
DEFECT_ROOT = 'C:\\Users\\Lenovo\\PCB Python\\images'     
OUTPUT_DIR = 'C:\\Users\\Lenovo\\PCB Python\\processed_masks_and_aligned_images' 


def align_and_subtract_grayscale(template_path, test_path, max_features=500, good_match_percent=0.15):

    #Load Images in Grayscale
    template_gray = cv2.imread(template_path, 0)
    test_gray = cv2.imread(test_path, 0)
    
    if template_gray is None or test_gray is None:
        raise FileNotFoundError(f"Could not load images. Template: {template_path}, Test: {test_path}")

    #Image alinment    
    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(template_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(test_gray, None)

    if descriptors1 is None or descriptors2 is None:
        raise ValueError("Could not find enough features for alignment.")
        
   #Use Brute-Force Matcher
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)

    #creating a list to store good matches
    matches_list = list(matches) 
    matches_list.sort(key=lambda x: x.distance)

    num_good_matches = int(len(matches_list) * good_match_percent)
    good_matches = matches_list[:num_good_matches]
    
    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    h, _ = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

    if h is None:
        raise ValueError("Homography calculation failed. Alignment not possible.")
                         
    height, width = template_gray.shape
    aligned_test_image = cv2.warpPerspective(test_gray, h, (width, height))

    #Image Subtraction
    difference_map = cv2.absdiff(template_gray, aligned_test_image)
    #Otsu Thresholding
    _, defect_mask = cv2.threshold(difference_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    
    return aligned_test_image, defect_mask

def process_structured_dataset_module1(template_root, defect_root, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    total_processed = 0
    
    # Outer Loop to go through the defect type folders 'missing holes' etc.
    for defect_type_folder in os.listdir(defect_root):
        defect_type_path = os.path.join(defect_root, defect_type_folder)
        
        if not os.path.isdir(defect_type_path):
            continue

        print(f"\n--- Processing Defect Type: {defect_type_folder} ---")
        
        # Nested loop to go through the circuit ID folders 01,02 etc.
        for template_id_folder in os.listdir(defect_type_path):
            template_id_path = os.path.join(defect_type_path, template_id_folder)
            
            if not os.path.isdir(template_id_path):
                continue

            template_id = template_id_folder
            #golden PCBs - '01.jpg', '02.jpg', etc. in PCB_USED
            template_file_name = f"{template_id}.jpg"
            template_path = os.path.join(template_root, template_file_name)
            
            if not os.path.exists(template_path):
                print(f"!! ERROR: Golden PCB not found: {template_path}. Skipping {template_id_folder}.")
                continue
            
            # Inner Loop to go through all test images '01_missing_hole_01.jpg' etc.
            test_images = [f for f in os.listdir(template_id_path) if f.lower().endswith(('.jpg', '.png'))]
            
            if not test_images:
                continue
                
            # Creating output folder 
            output_sub_dir = os.path.join(output_dir, defect_type_folder, template_id_folder)
            os.makedirs(output_sub_dir, exist_ok=True)
            
            for test_file in test_images:
                test_path = os.path.join(template_id_path, test_file)
                
                try:
                    aligned_img, defect_mask = align_and_subtract_grayscale(template_path, test_path)

                    # Save Output
                    base_name = os.path.splitext(test_file)[0]
                    
                    # Save Aligned Image
                    aligned_path = os.path.join(output_sub_dir, f'{base_name}_aligned.jpg')
                    cv2.imwrite(aligned_path, aligned_img)

                    # Save Defect Mask
                    mask_path = os.path.join(output_sub_dir, f'{base_name}_mask.png')
                    cv2.imwrite(mask_path, defect_mask)
                    
                    total_processed += 1
                
                except Exception as e:
                    print(f"ERROR processing {test_file}: {e}")

    print(f"\n preprocessing finished. Total pairs processed: {total_processed}")

#EXECUTION

if __name__ == "__main__":
    if TEMPLATE_ROOT.startswith('path/to/') or DEFECT_ROOT.startswith('path/to/'):
        print("!!! PLEASE UPDATE TEMPLATE_ROOT AND DEFECT_ROOT PATHS !!!")
    else:
        process_structured_dataset_module1(TEMPLATE_ROOT, DEFECT_ROOT, OUTPUT_DIR)