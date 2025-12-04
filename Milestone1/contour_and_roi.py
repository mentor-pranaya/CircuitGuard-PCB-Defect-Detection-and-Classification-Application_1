#In this program we have created a seprate folders for Contours and ROIs in which we process and store them
import cv2
import os
errors = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
pad = 5
min_area = 50 
min_size = 10

#Creating folders for countour (if not already there)
for error_type in range(6):
     os.makedirs(f"Output_data/contour/{errors[error_type]}", exist_ok=True)

#Adding Contours to all Images
for error_type in range(6):
     for i in range(1,13):
         for j in range(1,21):
                 img1 = cv2.imread(f"Output_data/Subtraction/{errors[error_type]}/{i:02d}_{errors[error_type]}_{j:02d}.jpg", cv2.IMREAD_GRAYSCALE)
                 if img1 is None:
                     print("Changing PCB")
                     break
                 img2 = cv2.imread(f"Output_data/BW_images/{errors[error_type]}/{i:02d}_{errors[error_type]}_{j:02d}.jpg")
                 contours1, heirarchy1 = cv2.findContours(img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                 img_contours = cv2.drawContours(img2.copy(), contours1, -1, (255,0,0), 2)
                 cv2.imwrite(f"Output_data/contour/{errors[error_type]}/{i:02d}_{errors[error_type]}_{j:02d}.jpg", img_contours)

                 #Folder for each image to store multiple rois
                 os.makedirs(f"Output_data/ROI/{errors[error_type]}/{i:02d}_{errors[error_type]}_{j:02d}", exist_ok=True)

                 #roi extraction
                 for a,cnt in enumerate(contours1):
                     area = cv2.contourArea(cnt)
                     if area < min_area:
                        continue
                     
                     x,y,w,h = cv2.boundingRect(cnt)

                     if w < min_size or h < min_size:
                         continue
                     
                     x_start = max(x - pad, 0)
                     y_start = max(y - pad, 0)
                     x_end = min(x + pad + w, img2.shape[1])
                     y_end = min(y + pad + h, img2.shape[0])
                     roi = img2[y_start:y_end, x_start:x_end]

                     cv2.imwrite(f"Output_data/ROI/{errors[error_type]}/{i:02d}_{errors[error_type]}_{j:02d}/{i:02d}_{errors[error_type]}_{j:02d}_roi_{a}.jpg",roi)
