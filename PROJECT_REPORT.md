## CircuitGuard: PCB Defect Detection and Classification Application

# Technical Project Report

# Summary
CircuitGuard is an automated defect detection and classification system for Printed Circuit Boards (PCBs) developed using deep learning and computer vision techniques. The system leverages EfficientNet-B4 architecture to classify six types of PCB defects with 97.76% accuracy, exceeding the target requirement of 97%.
# Key Achievements
Trained EfficientNet-B4 model achieving 97.76% test accuracy
Developed functional web application with real-time inference
# Requirements
Detect and localize defects accurately
Classify defects into specific categories
Achieve ≥97% classification accuracy
Provide results in < 5 seconds per image
Offer user-friendly interface for operators
Generate exportable reports and logs
# Objectives
Defect Detection: Identify defect locations using reference-based image subtraction
Defect Classification: Classify defects into six categories using deep learning
High Accuracy: Achieve ≥97% classification accuracy on test set
Application Development: Build web-based interface
# Dataset
PCB_DATASET/
├── PCB_USED/           # 10 template (defect-free) images
├── images/             # defect images organized by type
│   ├── missing_hole/    
│   ├── mouse_bite/     
│   ├── open_circuit/   
│   ├── short/          
│   ├── spur/           
│   └── spurious_copper/# 116 images
├── Annotations/        # Bounding box annotations (XML)
└── rotation/           # Augmented rotated versions
# Dataset Defect Catagory
Missing Hole - Absent drilled holes where connections should exist
Mouse Bite - Small notches or irregularities on board edges
Open Circuit - Incomplete electrical pathways or broken traces 
Short Circuit - Unintended connections between conductors 
Spur - Unwanted copper protrusions extending from traces
Spurious Copper - Extra copper deposits in unintended areas
# Data Preprocessing
From the original dataset, 2,953 ROI patches (128×128 pixels) were extracted using bounding box annotations, resulting in:
Training Set: 2,061 samples (70%)
Validation Set: 446 samples (15%)
Test Set: 446 samples (15%)
#  Overall Pipeline
Input Image → Preprocessing → Defect Detection → ROI Extraction → 
Classification → Results Display → Export

# Module 1: Data Preparation
Dataset loading and verification
Statistical analysis
XML annotation parsing
Template-defect image mapping
Visualization generation

# Module 2: ROI Extraction
Contour detection using OpenCV
Bounding box extraction from annotations
ROI cropping (128×128 patches)
Dataset organization (train/val/test split)

# Module 3: Model Training
EfficientNet-B4 architecture implementation
Data augmentation pipeline
Training with Adam optimizer
Validation and early stopping
Model checkpointing

# Module 4: Evaluation and Testing
Test set inference
Confusion matrix generation
Per-class metrics calculation
Error analysis
Confidence distribution analysis
# Libraries:
Deep Learning: PyTorch 2.0.1, timm
Computer Vision: OpenCV 4.8.0, PIL
Web Framework: Flask 3.0.0, Streamlit 1.28.0
Data Processing: NumPy, Pandas, scikit-learn

#  Web Application Architecture
Drag-and-drop image upload
Real-time preview
Animated result display
Responsive design
# Training Results
Total Training Time: 26.08 minutes
Epochs Trained: 34 (early stopped)
Best Validation Accuracy: 97.31%
Final Test Accuracy: 97.76% 
# Overall Metrics:
Test Accuracy: 97.76%
Precision: 0.9776
Recall: 0.9776
F1-Score: 0.9775
# "Spur" class showed highest performance (98.5% F1)
# Confusion Matrix Analysis
Main Diagonal Strength:
97.76% of predictions on diagonal (correct)
Strong discriminative features learned
Minor Confusions:
Short ↔ Open Circuit: 2 cases (visually similar)
Spur ↔ Spurious Copper: 1 case (both copper-related)

# Export Options
Mode 1: Classification Only
Mode 2: Full Pipeline

# Model Improvements in future:
Ensemble Methods: Combine multiple models for voting-based prediction
Advanced Architectures: Explore Vision Transformers (ViT) for comparison
Semi-Supervised Learning: Utilize unlabeled data for performance boost

                                                                                                End of Report



