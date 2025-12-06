# 📌 PCB Defect Detection & Classification System

A complete automated pipeline for detecting and classifying PCB (Printed Circuit Board) defects using **Computer Vision**, **Image Processing**, and **Deep Learning (EfficientNet-B4)**.  
The system performs template matching, alignment, defect mask generation, ROI extraction, classification, visualization, and export through a **Streamlit UI**.

---

## 🚀 Features
- Automatic template matching (SSIM)
- ORB-based image alignment using Homography (RANSAC)
- Pixel-wise difference map (absdiff)
- Otsu thresholding + morphological refinement
- Contour-based ROI extraction
- EfficientNet-B4 classifier for 6 defect types
- Streamlit UI for upload, visualization, and exporting
- Auto-saved annotated images + CSV defect logs
- Fully modular and industry-ready pipeline

---

## 📂 Project Structure

PCB-Defect-Detection/
│── app.py # Streamlit UI
│── inference.py # Backend inference engine
│── module1_alignment.py
│── module2_roi_extraction.py
│── module3_training.py
│── module4_evaluation.py
│── module6_backend.py
│── module7_export.py
│── requirements.txt
│── README.md
│
├── PCB_USED/ # Defect-free templates
├── images/ # Labeled defect images
├── masks_cleaned/ # Output of Module 1
├── ROIs/ # Output of Module 2
├── module7_outputs/ # Annotated images + CSV logs


---

## 🧠 Supported Defect Classes
- Missing_hole  
- Mouse_bite  
- Open_circuit  
- Short  
- Spur  
- Spurious_copper  

---

▶️ Running the Web App
streamlit run app.py


Upload a PCB defect image and the system will automatically:
-✓ Match with best template
-✓ Align using ORB + Homography
-✓ Generate difference map
-✓ Produce defect mask
-✓ Extract ROIs
-✓ Classify each defect
-✓ Display annotated image
-✓ Export CSV log + annotated output

🖼️ Pipeline Overview

Template Matching → SSIM
Alignment → ORB + RANSAC Homography
Subtraction → cv2.absdiff
Thresholding → Otsu
Mask Cleanup → Morphology
Contour Extraction → ROI
EfficientNet Classification
UI Visualization + Export

📊 Model Performance

High accuracy on unseen ROIs
Good generalization across all 6 defect categories
Supports modular retraining with new data
(Add your training/validation curves if needed.)

🧪 Technologies Used

Python
OpenCV
PyTorch (EfficientNet-B4)
NumPy, Pandas
Scikit-Image
Matplotlib, Seaborn
Streamlit

🔮 Future Enhancements

YOLO-based real-time defect detection
Better rotation/perspective correction
Unsupervised defect segmentation
Integration with factory inspection cameras

