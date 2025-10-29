import os
import cv2
import numpy as np

TEMPLATE_ROOT = 'PCB_USED' 
DEFECT_ROOT = 'images'     
OUTPUT_DIR = 'Result' 

def align_and_subtract_grayscale(template_path, test_path, max_features=500, good_match_percent=0.15):
    template_gray = cv2.imread(template_path, 0)
    test_gray = cv2.imread(test_path, 0)
    
    if template_gray is None or test_gray is None:
        raise FileNotFoundError(f"Could not load images. Template: {template_path}, Test: {test_path}")

    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(template_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(test_gray, None)

    if descriptors1 is None or descriptors2 is None:
        raise ValueError("Could not find enough features for alignment.")
        
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)

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

    difference_map = cv2.absdiff(template_gray, aligned_test_image)
    _, defect_mask = cv2.threshold(difference_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    
    return aligned_test_image, defect_mask

def process_pcb_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_processed = 0
    
    defect_types = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
    
    template_path = os.path.join(TEMPLATE_ROOT, '01.JPG')
    
    if not os.path.exists(template_path):
        print(f"ERROR: Template image not found: {template_path}")
        return
    
    print(f"Using template: {template_path}")
    
    for defect_type in defect_types:
        defect_path = os.path.join(DEFECT_ROOT, defect_type)
        
        if not os.path.exists(defect_path):
            print(f"Warning: Defect folder not found: {defect_path}")
            continue

        print(f"Processing Defect Type: {defect_type}")
        
        test_images = [f for f in os.listdir(defect_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not test_images:
            print(f"No images found in {defect_type}")
            continue
            
        output_sub_dir = os.path.join(OUTPUT_DIR, defect_type)
        os.makedirs(output_sub_dir, exist_ok=True)
        
        for test_file in test_images:
            test_path = os.path.join(defect_path, test_file)
            
            try:
                print(f"Processing: {test_file}")
                aligned_img, defect_mask = align_and_subtract_grayscale(template_path, test_path)

                base_name = os.path.splitext(test_file)[0]
                
                aligned_path = os.path.join(output_sub_dir, f'{base_name}_aligned.jpg')
                cv2.imwrite(aligned_path, aligned_img)

                mask_path = os.path.join(output_sub_dir, f'{base_name}_mask.png')
                cv2.imwrite(mask_path, defect_mask)
                
                total_processed += 1
                print(f"Saved: {base_name}_aligned.jpg and {base_name}_mask.png")
                
            except Exception as e:
                print(f"ERROR processing {test_file}: {e}")

    print(f"Processing finished. Total images processed: {total_processed}")
    print(f"Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    print("Starting PCB Image Subtraction Pipeline")
    process_pcb_images()
