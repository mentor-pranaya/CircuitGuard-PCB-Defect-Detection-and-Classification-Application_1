#In this program we have created a seprate folder in which we process and store the B&W images all at once
import cv2
import os
errors = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]

#Creating folders (if not already there)
for error_type in range(6):
    os.makedirs(f"Output_data/BW_images/{errors[error_type]}", exist_ok=True)
os.makedirs("Output_data/BW_images/PCB_USED", exist_ok=True)

#converting the golden pcbs to B&W
for pcb_type in range(1,13):
    try:
        img = cv2.imread(f"PCB_DATASET/PCB_USED/{pcb_type:02d}.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"Output_data/BW_images/PCB_USED/{pcb_type:02d}.jpg", img)

    except cv2.error:
        print("Changing PCB")
        continue

#Converting all images from color to B&W and saving them
flag = False
for error_type in range(6):
    for i in range(1,13):
        for j in range(1,21):
            try:
                img = cv2.imread(f"PCB_DATASET/images/{errors[error_type]}/{i:02d}_{errors[error_type]}_{j:02d}.jpg")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # cv2.imshow("image", img)
                cv2.imwrite(f"Output_data/BW_images/{errors[error_type]}/{i:02d}_{errors[error_type]}_{j:02d}.jpg", img)
            except cv2.error:
                print("Changing PCB")
                flag = True
            if flag:
                flag = False
                break
