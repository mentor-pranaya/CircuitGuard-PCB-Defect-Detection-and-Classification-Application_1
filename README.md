## CircuitGuard - PCB Defect Detection and Classification :- An automated defect detection and classification system for Printed Circuit Boards (PCBs)

# Overview
CircuitGuard is a comprehensive PCB defect detection system that combines:
Reference-based image subtraction to identify defect regions
Contour detection to extract regions of interest
EfficientNet-B4 deep learning model for accurate defect classification
Interactive web interface for real-time prediction and visualization

# The system can detect and classify 6 types of PCB defects:
Missing Hole
Mouse Bite
Open Circuit
Short
Spur
Spurious Copper

# Features
Two Detection Modes:
Classification Only: Direct classification of pre-cropped defect images
Full Pipeline: Complete workflow with template comparison, defect detection, and classification

# Key Capabilities:

✅ Upload and process PCB images
✅ Real-time defect detection
✅ Confidence scores for each prediction
✅ Top-3 prediction probabilities
✅ Export annotated images
✅ Export prediction logs as CSV
✅ Interactive and responsive UI

# Tech Stack
PyTorch, timm (EfficientNet-B4)
OpenCV, NumPy
Streamlit(Web Framework)
Adam Optimizer, Cross-Entropy Loss(Training)
Pandas, PIL(Data Processing)

# Install Dependencies:
pip install streamlit torch torchvision timm opencv-python pillow numpy

# Download Model:
Place the trained model file efficientnet_state_dict.pkl in the main folder in which app.py exists.

# Run the Application:
streamlit run app.py
The app will open in your browser at http://localhost:8501

# Usage
Classification Only Mode:
Select "Classification Only" in the sidebar
Upload a PCB defect image
Click "🔍 Classify Defect"
View prediction results with confidence scores
Download results using export buttons

Full Pipeline Mode:
Select "Full Pipeline (with Template)" in the sidebar
Upload Template Image (defect-free PCB)
Upload Test Image (PCB with defects)
Click "🔍 Detect Defects"
View:
Difference maps
Threshold visualization
Detected regions with bounding boxes
Classification for each region

# Download annotated images and prediction logs

# Model Performance
Training Results (Module 4)

Model: EfficientNet-B4
Training Epochs: 24
Best Validation Accuracy: 97.31%
Test Accuracy: 97.76% ✅ (Target: ≥97%)


## Author
Suswagata ghosh



