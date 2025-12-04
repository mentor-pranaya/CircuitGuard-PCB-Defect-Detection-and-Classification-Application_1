#In this program we have created a seprate folders for Subtracted images
import cv2
from matplotlib import pyplot as plt
import os

errors = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]

#Creating folders for subtracted images (if not already there)
for error_type in range(6):
    os.makedirs(f"Output_data/Subtraction/{errors[error_type]}", exist_ok=True)

flag = False
for pcb_type in range(1,13):
    #Reading golden PCB image to subtract from
    img1 = cv2.imread(f"Output_data/BW_images/PCB_USED/{pcb_type:02d}.jpg")
    if img1 is None:
        print(f"changing pcb as {pcb_type:02d}.jpg not found")
        continue
    for error_type in range(6):
            for j in range(1,21):
                #reading test pcb images
                img2 = cv2.imread(f"Output_data/BW_images/{errors[error_type]}/{pcb_type:02d}_{errors[error_type]}_{j:02d}.jpg")
                if img2 is None:
                    print(f"changing pcb as {pcb_type:02d}_{errors[error_type]}_{j:02d}.jpg not found")
                    break
                #Subtracting the images
                subtracted_img = cv2.absdiff(img1,img2)
                #Thresholding the images in binary
                _, thresh1 = cv2.threshold(subtracted_img, 20, 255, cv2.THRESH_BINARY)
                cv2.imwrite(f"Output_data/Subtraction/{errors[error_type]}/{pcb_type:02d}_{errors[error_type]}_{j:02d}.jpg", thresh1)

#Display Sample images
original_img = cv2.imread("Output_data/BW_images/missing_hole/01_missing_hole_01.jpg")
subtracted_img = cv2.imread("Output_data/Subtraction/missing_hole/01_missing_hole_01.jpg")

# Plot side by side
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(subtracted_img)
plt.title("Subtracted Image")
plt.axis("off")

plt.tight_layout()
plt.show()
