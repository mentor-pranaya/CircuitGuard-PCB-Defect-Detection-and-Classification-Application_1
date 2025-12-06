📌 PCB Defect Detection & Classification System

A complete end-to-end automated pipeline for detecting and classifying PCB (Printed Circuit Board) defects using Computer Vision, Image Processing, and Deep Learning (EfficientNet-B4).
The system performs template matching, image alignment, defect ROI extraction, classification, visualization, and result exporting through a user-friendly Streamlit UI.

🚀 Features

✔ Automatic template matching using SSIM
✔ ORB-based image alignment (Homography + RANSAC)
✔ Pixel-wise difference map + Otsu thresholding
✔ Contour-based ROI extraction
✔ EfficientNet-B4 classifier for 6 defect types
✔ Streamlit-based UI for visualization
✔ Auto-generated annotated images
✔ CSV defect logs (bounding box, label, confidence, timestamp)
✔ Industry-ready end-to-end pipeline

Project Structure
PCB-Defect-Detection/
│── inference.py
│── app.py                   # Streamlit UI
│── module1_alignment.py
│── module2_roi_extraction.py
│── module3_training.py
│── module4_evaluation.py
│── module6_backend.py
│── module7_export.py
│── README.md
│
├── PCB_USED/                # Template defect-free images
├── images/                  # Defect images organized by class
├── masks_cleaned/           # Output of Module 1
├── ROIs/                    # Output of Module 2
├── module7_outputs/         # Annotated images + logs

🧠 Supported Defect Classes
Missing_hole
Mouse_bite
Open_circuit
Short
Spur
Spurious_copper

▶️ Running the Web App (Streamlit UI)
streamlit run app.py


Upload any PCB defect image and the system will automatically:
✔ Match template
✔ Align the image
✔ Generate difference map
✔ Extract ROIs
✔ Classify defects
✔ Visualize results
✔ Export annotated image + CSV log

🖼️ Pipeline Overview
Template Matching
SSIM selects the best matching defect-free PCB.
Image Alignment
ORB keypoints + RANSAC Homography.
Difference Computation
cv2.absdiff() → defect heatmap.
Mask Generation
Otsu threshold + morphology cleanup.
ROI Extraction
Contours → bounding boxes.
Classification
EfficientNet-B4 predicts defect category.
Visualization & Export
Annotated images & CSV logs saved automatically.

📊 Model Performance
High accuracy on defect ROIs
Strong generalization on unseen PCB samples
Balanced performance across all 6 classes

🧪 Technologies Used
Python + OpenCV
PyTorch (EfficientNet-B4)
Scikit-Image
NumPy, Pandas
Streamlit
Matplotlib, Seaborn

🔮 Future Improvements
Improve rotation & perspective robustness
Add YOLO-based detection pipeline
Support for more defect types
Real-time video stream inference
