import os
import cv2
import numpy as np
import glob

# folder in which image subtraction outputs are saved
PROCESSED_ROOT = 'C:\\Users\\Lenovo\\PCB Python\\processed_masks_and_aligned_images1' 

#folder in which cropped ROI images will be saved
ROI_OUTPUT_ROOT = 'C:\\Users\\Lenovo\\PCB Python\\final_defect_roi_1' 
MIN_DEFECT_AREA = 100 

#ROI EXTRACTION

def extract_defects_and_save(aligned_image, defect_mask, output_sub_dir, base_name, defect_type, min_area=200):

    kernel= np.ones((5,5), np.uint8) 
    
    mask_open = cv2.erode(defect_mask, kernel, iterations=1)
    mask_open = cv2.dilate(mask_open, kernel, iterations=1)

    dilated_mask = cv2.dilate(mask_open, kernel, iterations=1)
    cleaned_mask = cv2.erode(dilated_mask, kernel, iterations=1)
    
    #Contour Detection
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    roi_count = 0
    
    #Bounding Box Generation and ROI Extraction
    for cnt in contours:
        # Filter small noise contours
        if cv2.contourArea(cnt) < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        
        # 4. Extract ROI and Save
        roi_patch = aligned_image[y:y+h, x:x+w]
        
        #naming format for cropped images
        defect_label = defect_type.replace(' ', '_') 
        roi_filename = f'{base_name}_{defect_label}_{roi_count}.jpg'
        
        #saving the ROI patch to the output folder
        cv2.imwrite(os.path.join(output_sub_dir, roi_filename), roi_patch)
        
        roi_count += 1
        
    return roi_count

def run_pipeline(processed_root, roi_output_root):
    
    print(f"Starting Module 2: Processing masks in {processed_root}")
    total_rois_extracted = 0

    mask_files = glob.glob(os.path.join(processed_root, '**/*_mask.png'), recursive=True)
    
    if not mask_files:
        print("ERROR: No *_mask.png files found. Ensure Module 1 was run correctly and PROCESSED_ROOT is correct.")
        return
        
    for mask_path in mask_files:
        try:

            parts = mask_path.split(os.path.sep)
            root_index = parts.index(os.path.basename(processed_root.rstrip(os.path.sep)))
            
            defect_type = parts[root_index + 1]
            template_id_folder = parts[root_index + 2]
            base_name = os.path.basename(mask_path).replace('_mask.png', '')
            aligned_filename = f"{base_name}_aligned.jpg"
            aligned_path = os.path.join(os.path.dirname(mask_path), aligned_filename)
            output_sub_dir = os.path.join(roi_output_root, defect_type, template_id_folder)
            os.makedirs(output_sub_dir, exist_ok=True)
            defect_mask = cv2.imread(mask_path, 0)
            aligned_img = cv2.imread(aligned_path, 0)

            if aligned_img is None or defect_mask is None:
                print(f"Warning: Could not load paired images for {mask_path}. Skipping.")
                continue

            #Extracting ROIs
            roi_count = extract_defects_and_save(
                aligned_img, 
                defect_mask, 
                output_sub_dir, 
                base_name, 
                defect_type, 
                MIN_DEFECT_AREA
            )
            
            total_rois_extracted += roi_count
            print(f"Processed {os.path.basename(mask_path)}: Extracted {roi_count} ROIs.")

        except Exception as e:
            print(f"Error processing {mask_path}: {e}")
            
    print(f"\n Total ROIs extracted: {total_rois_extracted}")


#EXECUTION

if __name__ == "__main__":
    if PROCESSED_ROOT.startswith('path/to/'):
        print("!!! PLEASE UPDATE PROCESSED_ROOT AND ROI_OUTPUT_ROOT PATHS !!!")
    else:
        run_pipeline(PROCESSED_ROOT, ROI_OUTPUT_ROOT)