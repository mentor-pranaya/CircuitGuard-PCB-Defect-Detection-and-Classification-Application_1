# CircuitGuard-PCB-Defect-Detection-and-Classification-Application_1 

# CircuitGuard: PCB Defect Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Enterprise-FF4B4B)
![PyTorch](https://img.shields.io/badge/PyTorch-EfficientNet-EE4C2C)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

CircuitGuard is an industrial-grade Automated Optical Inspection (AOI) tool powered by Deep Learning. It leverages a fine-tuned EfficientNet-B4 architecture to detect and classify manufacturing defects in Printed Circuit Boards (PCBs) with high precision, providing real-time diagnostics and audit-ready data logs.

---

## Key Features

* **Advanced AI Core**: Utilizes EfficientNet-B4 via Transfer Learning for high-accuracy defect classification.
* **Real-Time Inference**: Optimized backend pipeline provides instant analysis of PCB images.
* **Visual Diagnostics**: Generates annotated defect maps with confidence heatmaps and localization boxes.
* **Enterprise Dashboard**: Professional UI designed to reduce operator fatigue and increase data readability.
* **Audit Trails**:
    * **Session History**: Tracks all scans performed in a session.
    * **CSV Export**: One-click download of detailed inspection logs.
    * **Image Export**: Download annotated evidence images for reporting.
* **Robust Architecture**: Includes crash-proof rendering and automated error handling for older environments.

---

## Supported Defect Classes

The model is trained to identify the following 6 PCB defect types:

1.  **Missing Hole** (Structural error)
2.  **Mouse Bite** (Edge irregularity)
3.  **Open Circuit** (Connectivity break)
4.  **Short** (Unintended connectivity)
5.  **Spur** (Copper protrusion)
6.  **Spurious Copper** (Excess material)

---

## Technical Stack

* **Frontend**: Streamlit (Python web framework)
* **Backend**: PyTorch (Inference engine), Timm (Model architecture)
* **Image Processing**: PIL (Pillow), NumPy, Torchvision
* **Data Handling**: Pandas (Session logging and CSV export)

---

## Project Structure

```text
CircuitGuard/
├── app.py                     # Main Application Entry Point (Frontend)
├── backend_inference.py       # Core Logic: Preprocessing & Inference
├── best_efficientnet_b4.pth   # Trained Model Weights (Required)
├── class_mapping.json         # JSON mapping of Class IDs to Names
├── requirements.txt           # Python Dependencies
└── README.md                  # Project Documentation
