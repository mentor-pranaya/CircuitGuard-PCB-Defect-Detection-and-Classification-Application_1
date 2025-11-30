# CircuitGuard-PCB-Defect-Detection-and-Classification-Application_1 

CircuitGuard is a complete AI system designed to automatically detect defects in Printed Circuit Boards (PCBs) using image processing + deep learning.
It integrates:
Template subtraction
ROI extraction
EfficientNet-based classification
Batch inference
Streamlit UI
TorchScript deployment
This project strictly follows the structure and methodology described in the CircuitGuard PDF.

CircuitGuard_Project/
│
├── app.py                                # Streamlit App (Module 05)
├── module06_backend.py                   # Backend inference engine (Module 06)
│
├── Module 01.ipynb                       # Image subtraction + preprocessing
├── Module 02.ipynb                       # ROI extraction + mask generation
├── Module 03.ipynb                       # Model training (EfficientNet-B0 CPU)
├── Module 04.ipynb                       # Model evaluation on 693 images
├── Module 05.ipynb                       # Streamlit integration & testing
├── Module 06.ipynb                       # Batch inference pipeline
├── Module 07.ipynb                       # TorchScript export + final summary
│
├── selected_paths.json                   # Auto-detected template/test paths
│
├── checkpoints/                          # Trained models
│   ├── best_effnet_b0_cpu.pth
│   ├── best_effnet_b0_cpu_ultrafast.pth
│   └── effnet_b0_cpu_traced.pt           # TorchScript (Module 07)
│
├── inference_results/                    # Streamlit predictions
│
├── inference_results_module06_test/      # Module 06 single-pair inference
│
├── inference_results_module06_batch_fast/ # Module 06 batch inference
│
├── module07_evaluation/                  # Final evaluation metrics
│   ├── summary_report.csv
│   ├── confusion_matrix.png
│   └── per_class_results.csv
│
└── README.md

Model Used
✔ EfficientNet-B0 (pretrained)
Optimized for CPU with:
Mixed Precision disabled for CPU
AdamW optimizer
CrossEntropy with class weights
FastDataloader (num_workers=0)
Ultrafast training mode (reduced epochs)
Supported PCB Defects

The system detects 6 PCB defect types:
Missing Hole
Mouse Bite
Open Circuit
Short
Spur
Spurious Copper

Module 01 — Preprocessing & Template Subtraction
Loads template & test image
Converts to grayscale
Performs absolute subtraction
Otsu thresholding
Morphological operations
Saves masks and previews
Outputs ROI candidates
Main outputs:
Subtraction image
Binary mask
ROI bounding boxes on PCB

Module 02 — ROI Extraction
Reads mask
Detects contours
Extracts bounding-box crops
Visualizes them
Saves ROI patches
Output:
Individual ROI images
JSON map of selected template/test pair

Module 03 — Model Training
Trains EfficientNet-B0 on extracted ROI dataset.
Features:
CPU optimized training loop
Progress bar
Best checkpoint saving
Ultrafast mode (2–3 minutes training)
Class-weighted loss
ReduceLROnPlateau scheduler
Outputs saved in:
checkpoints/
   best_effnet_b0_cpu.pth
   best_effnet_b0_cpu_ultrafast.pth

Module 04 — Model Evaluation
Evaluates the trained model on 693 rotated PCB images.
Generates:
Predictions for every ROI
Full classification report
Per-defect confusion matrix
CSV logs of all detections
Saved to:
module07_evaluation/
and
inference_results/

Module 05 — Streamlit Web App
Allows users to upload a PCB image and detect defects from browser.
Run using:
streamlit run app.py
Features:
File upload UI
Runs model inference
Shows prediction boxes on PCB
Scores and defect labels
Saves results in inference_results/

Module 06 — Backend Pipeline
The engine used by Streamlit or batch processing.
Includes:
module06_backend.py
Single-pair inference
Batch inference for whole dataset
Annotation + CSV export
Outputs saved to:
inference_results_module06_test/
inference_results_module06_batch_fast/

Module 07 — Model Export (TorchScript)
Converts trained EfficientNet model to:
TorchScript .pt file
Faster loading
Deployment-ready
Saved as:
checkpoints/effnet_b0_cpu_traced.pt
Also generates:
Confusion matrix plot
Summary CSV
Per-class metrics
Saved in:
module07_evaluation/

How to Run the Project
🟩 1. Install Requirements
Activate environment:
conda activate circuitguard
pip install -r requirements.txt   (optional)
pip install torch torchvision timm opencv-python-headless numpy pillow matplotlib seaborn streamlit scikit-learn tqdm

2. Run Streamlit App
cd CircuitGuard_Project
streamlit run app.py

4. Run Module Notebooks
Open Jupyter Notebook:
jupyter notebook
Then run each:
Module 01.ipynb
Module 02.ipynb
…
Module 07.ipynb

Your final model may give results like:

Class	Precision	Recall	F1
Spur	0.88	0.87	0.88
Mouse Bite	0.72	0.68	0.70
Short	0.74	0.74	0.74
Missing Hole	0.32	0.31	0.31
Open Circuit	0.52	0.53	0.52
Spurious Copper	0.71	0.72	0.71

(Values will vary based on your training).

