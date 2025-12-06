# CircuitGuard-PCB-Defect-Detection-and-Classification-Application_1 
CircuitGuard is an end-to-end automated system designed to detect and classify defects in Printed Circuit Boards (PCBs) using a combination of traditional computer vision techniques and deep learning.
The system includes a complete preprocessing pipeline, a trained EfficientNet-B4 classifier, a Flask backend for inference, and a modern web-based frontend for user interaction.

## 1. Overview
Printed Circuit Boards are central to modern electronics, and even minor defects can lead to system failures, increased production costs, or device malfunction.
This project provides a scalable software-based alternative to expensive Automated Optical Inspection (AOI) systems by:

* Automatically identifying structural differences between a test PCB and its golden reference
* Extracting defect regions using contour detection
* Classifying defects using a trained deep learning model
* Providing an interactive web interface for visualization and reporting

The system supports detection of the following defect types:

1. Missing Hole
2. Mouse Bite
3. Open Circuit
4. Short
5. Spur
6. Spurious Copper

## 2. Key Features
### Automated Defect Detection
The system performs golden–test subtraction and contour-based extraction to locate defective regions.

### Deep Learning Classification
An EfficientNet-B4 model trained on defect ROI images achieves high accuracy and provides confidence scores for each prediction.

### Automatic Golden PCB Identification
When multiple PCB designs exist, the system uses ORB feature matching to determine the correct golden reference image without requiring manual selection.

### Web-Based User Interface
Users can upload PCB images, run detections, view results, and download CSV/JSON reports through a clean HTML/CSS/JavaScript interface.

### Backend Processing Pipeline
Implemented in Flask, the backend performs image decoding, preprocessing, ROI extraction, model inference, and response generation.

## 3. Project Structure
'''bash
ProjectRoot/
│
├── Milestone1/                     # Initial image preprocessing and ROI extraction
│   ├── B&W.py                      # Grayscale + simple preprocessing tests
│   ├── contour_and_roi.py          # Contour detection & ROI extraction experiments
│   └── image_subtraction.py        # Golden–test image subtraction experiments
│
├── Milestone2/                     # Model training and evaluation workflow
│   ├── train_model.py              # EfficientNet-B4 training script
│   ├── train_test_split.py         # Dataset split into train/test folders
│   ├── pcb_defect_pipeline.py      # Early unified defect detection pipeline
│   ├── model_Evaluation.py         # Evaluation script producing metrics/reports
│   ├── Accuracy_curve.png          # Training accuracy plot
│   ├── loss_curve.png              # Training loss plot
│   └── confusion_matrix.png        # Final confusion matrix
│
├── Milestone3/                     # Early frontend/backend integration (Streamlit)
│   ├── app.py                      # Streamlit-based defect detection UI
│   └── index.html                  # Early HTML test UI
│
├── Milestone4/                     # Final production-ready deployment (Flask + HTML/CSS/JS)
│   ├── app.py                      # Flask server with full pipeline + API
│   ├── golden_pcb_detection.py     # ORB-based automatic golden PCB identification
│   ├── index.html                  # Final modern frontend UI
│   ├── pcb_defect_result.jpg       # Example annotated output image
│   └── golden/                     # Folder containing all golden reference PCBs
│       ├── golden1.jpg
│       ├── golden2.jpg
│       └── goldenN.jpg
│
├── efficientnet_b4_pcb.pth         # Trained EfficientNet-B4 model weights
│
├── README.md                       # Project documentation
└── LICENSE                         # Optional license file
'''


## 4. System Workflow
### Step 1: Golden PCB Identification (Optional)
If 'autoDetect' mode is selected, ORB feature matching compares the uploaded test PCB with all images inside the `golden/` folder to identify the correct golden reference.

### Step 2: Image Subtraction
Both images are converted to grayscale, and pixel-wise subtraction highlights structural differences.

### Step 3: Thresholding and Contour Extraction
Binary thresholding isolates difference regions.
Contours are extracted and filtered based on size and area to remove noise.

### Step 4: ROI Extraction
Bounding boxes around valid contours are expanded with padding.
Each cropped region becomes an ROI candidate.

### Step 5: Classification
ROIs are passed through the EfficientNet-B4 model, which outputs:

* Defect label
* Softmax-based confidence score

### Step 6: Output Generation
The final test PCB image is annotated with bounding boxes and defect labels.
Structured output is provided as:

* Annotated image (Base64)
* JSON containing defect metadata
* CSV report containing coordinates, bounding boxes, confidence and defect type

## 5. Model Details

### Architecture
* EfficientNet-B4
* Input size: 128 × 128
* Optimizer: Adam
* Loss function: CrossEntropy
* Output: Six defect classes

### Training
The model was trained on the Kaggle PCB Defects dataset, using cropped ROI images for each defect category.

### Performance
The final model achieved the following metrics on the test dataset:

* Accuracy: 98%
* Precision, Recall, F1-score: 0.97–1.00 across all classes
* Total evaluation samples: 2080

## 6. Running the Application
1. Download Milestone 4

Download the Milestone4 folder from the repository.
Your folder should contain:
* app.py
* index.html
* golden_pcb_detection.py
* golden/images
* efficientnet_b4_pcb.pth      ← add this model file
* requirements.txt

Make sure the golden/ folder contains all golden PCB reference images.

2. Install Requirements

Open a terminal inside the Milestone4 folder and install dependencies:

pip install -r requirements.txt

3. Start the Backend Server

Run:

python app.py


You should see:

Model loaded.
Golden PCBs loaded.
Server running at http://localhost:5000

Keep this terminal open.

4. Open the Frontend

Simply open index.html in your web browser.
No additional server is required for the UI.

5. Run PCB Defect Detection
   
Manual Mode
Upload Golden PCB
Upload Test PCB
Click Run Detection

Automatic Golden Identification Mode
Upload only the Test PCB

The system automatically selects the correct golden PCB from the golden/ folder
Defects are detected and classified

6. View & Export Results
You can download:

Annotated defect image
CSV defect log
JSON defect log

## 7. Notable Scripts

### pipeline.py
Implements the core detection and classification pipeline.
Contains key functions:

* `load_model`
* `subtract_images`
* `extract_contours`
* `classify_roi`
* `annotate_image`
* `run_pipeline`

### app.py
Handles:
* Base64 image decoding
* Golden PCB matching
* Preprocessing and classification
* Final result packaging
* API endpoint `/detect`

### Model_train.py
Script responsible for:
* Loading datasets
* Training EfficientNet-B4
* Saving model weights
* Logging metrics

## **8. Future Enhancements**
Potential improvements include:
* Real-time video inspection
* Alignment correction for shifted PCBs
* Full semantic segmentation of defect boundaries
* Deployment on embedded or edge devices
* Integration with factory automation workflows

## **10. Author**

Badal Dadwani
PCB Defect Detection Project — CircuitGuard
