<p align="center">
  <picture>
    <!-- Dark mode banner -->
    <source media="(prefers-color-scheme: dark)" srcset="assets/circuitguard_banner_dark.png">
    <!-- Light mode banner -->
    <source media="(prefers-color-scheme: light)" srcset="assets/circuitguard_banner_light.png">
    <!-- Fallback -->
    
  </picture>
</p>

<h1 align="center">🛡️ CircuitGuard</h1>
<h3 align="center">Automated PCB Defect Detection & Classification System</h3>

<p align="center">
  <a href="#-installation--setup">
    <img src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white" alt="Python">
  </a>
  <a href="#-tech-stack">
    <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
  </a>
  <a href="#-tech-stack">
    <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  </a>
  <a href="#-tech-stack">
    <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  </a>
</p>

---

## 📌 1. Project Overview

CircuitGuard is a fully automated **PCB (Printed Circuit Board) defect inspection system** combining traditional computer vision with deep learning.  

It performs:
- Image alignment and registration  
- Defect localization using image subtraction  
- ROI extraction for localized defects  
- EfficientNet-B4–based defect classification  
- Full-board annotation and overlay visualization  
- Web-based real-time PCB analysis  

**Supported defect classes (6):**
- Missing Hole  
- Mouse Bite  
- Open Circuit  
- Short  
- Spur  
- Spurious Copper  

The system exposes a user-friendly web UI where users can upload PCB images and receive annotated outputs, CSV defect logs, and confidence scores.

---

## 🧠 2. Key Features

- ✅ **Automatic Template Selection**  
  ORB-based feature matching to select the most appropriate defect-free template for any uploaded PCB.

- ✅ **Accurate Defect Localization**  
  Reference-based image subtraction followed by Otsu thresholding and morphological filtering to obtain clean defect masks.

- ✅ **High-Accuracy Classification**  
  EfficientNet-B4 model trained on a custom ROI dataset, achieving ≥97.6% accuracy on validation data.

- ✅ **Full Web Application**  
  Streamlit-based UI for real-time analysis, visualization, and user interaction.

- ✅ **Exports & Reporting**  
  - Annotated PCB image (PNG)  
  - Per-defect CSV prediction log  
  - Confidence scores and bounding boxes

- ✅ **Optimized Inference Pipeline**  
  - Model caching to avoid reloads  
  - Batch classification of ROIs  
  - Fallback mechanisms if alignment fails  

---

## 🛠️ 3. Tech Stack

### 🧩 Core Libraries

| Library        | Role                                   |
|----------------|----------------------------------------|
| Python 3.8+    | Core programming language              |
| OpenCV         | Image processing & feature extraction  |
| PyTorch        | Deep learning framework                |
| TIMM           | EfficientNet-B4 implementation         |
| NumPy / Pandas | Numerical ops & tabular logging        |
| Streamlit      | Web UI for inference and visualization |

### 🧠 ML Model

- **EfficientNet-B4** trained on automatically extracted ROIs from PCB images.  
- Input: 128×128 RGB defect patches.  
- Output: 6-way softmax over PCB defect classes.

### 👁️‍🗨️ Computer Vision

- ORB feature detection & matching  
- RANSAC-based homography estimation  
- Template-based image subtraction  
- Connected component labeling for ROI extraction  

---

## 🧩 4. System Architecture

```
flowchart LR
    I[User Upload PCB Image] 
    I --> II[Automatic Template Selection (ORB)]
    II --> III[Alignment (RANSAC Homography)]
    III --> IV[Image Subtraction]
    IV --> V[Otsu Thresholding]
    V --> VI[Morphological Filtering]
    VI --> VII[ROI Extraction (Connected Components)]
    VII --> VIII[EfficientNet-B4 Classification]
    VIII --> IX[Annotated PCB + CSV Log Output]
```

---

## 📦 5. Folder Structure

```
CircuitGuard/
│
├── Data/
│   ├── PCB_DATASET/
│   │   ├── templates/           # Defect-free PCB templates
│   │   ├── ROI_Dataset/         # 128x128 cropped training images
│   └── best_efficientnet_b4.pth # Trained EfficientNet-B4 weights
│
├── Module-1_Image_Subtraction_Otsu.ipynb
├── Module-2_Contour_ROI_Extraction.ipynb
├── Module-3_EfficientNet_Training.ipynb
├── Module-4_Inference.ipynb
├── Module-5_Pipeline.ipynb
├── module6_ui.py                  # Single-image ROI inference UI
├── Module-7(Complete_Application).py  # Full Streamlit application
│
├── README.md
└── requirements.txt
```

---

## 🔧 6. Installation & Setup

### 1️⃣ Clone the Repository

```
git clone https://github.com/YOUR_USERNAME/CircuitGuard.git
cd CircuitGuard
```

### 2️⃣ Create Virtual Environment (Recommended)

```
# Linux / macOS
python -m venv venv
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3️⃣ Install Requirements

```
pip install -r requirements.txt
```

### 4️⃣ Place Model Checkpoint

Place your trained EfficientNet-B4 checkpoint at:

```
Data/best_efficientnet_b4.pth
```

### 5️⃣ Add Template Images

Add defect-free PCB template images under:

```
Data/PCB_DATASET/templates/
```

---

## ▶️ 7. Run the Full Application

To launch the complete Streamlit web app:

```
streamlit run "Module-7(Complete_Application).py"
```

By default, the app will be available at:

```
http://localhost:8501
```

---

## ❓ 8. How the System Works

| Step | Stage                    | Description                                                                 |
|------|--------------------------|-----------------------------------------------------------------------------|
| 1    | Upload Image             | User uploads a PCB test image via the Streamlit UI.                        |
| 2    | Automatic Template Pick  | ORB-based matching selects the best defect-free template.                  |
| 3    | Alignment & Subtraction  | RANSAC aligns the uploaded PCB to the template, then subtraction highlights defects. |
| 4    | Mask Generation          | Otsu threshold + morphological closing produce a clean binary defect mask. |
| 5    | ROI Extraction           | Each connected defect component is cropped into a 128×128 ROI.            |
| 6    | EfficientNet Classification | Each ROI is classified with top-1/top-2 labels and confidence.        |
| 7    | Annotation & Export      | Final PCB image is annotated; CSV log with all detections is generated.    |

---

## ☑️ 9. Single-Image ROI Classification (Module 6)

To run the minimal UI for classifying a single ROI image:

```
streamlit run module6_ui.py
```

- Upload a 128×128 ROI image.  
- View top-1 and top-2 predicted classes along with confidence scores.

---

## 📁 10. Example Output Files

### 1. Annotated PCB Image

- PNG image with bounding boxes around all detected defects.  
- Each box labeled with predicted defect class and confidence.

### 2. CSV Log

Example column schema:

```
filename, template_used, x, y, w, h, top1_class, confidence, timestamp
```

---

## ℹ️ 11. Dataset Details

- Dataset is generated through ROI extraction using **Module 1** and **Module 2**.  
- Each ROI is standardized to **128 × 128 RGB**.  
- Contains **6 PCB defect classes**, balanced via data augmentation techniques.

---

## ⚠️ 12. Troubleshooting

- ❗ **Model not found?**  
  Ensure the checkpoint exists at:
  ```
  Data/best_efficientnet_b4.pth
  ```

- ❗ **Templates not loading?**  
  Check that your templates directory is correctly populated:
  ```
  Data/PCB_DATASET/templates/
  ```

- ❗ **Streamlit not running?**  
  Make sure dependencies are installed:
  ```
  pip install streamlit opencv-python torch timm pandas numpy pillow
  ```

---

## 🚀 13. Future Enhancements

- Replace ROI classification with **end-to-end YOLO-style object detection**  
- Containerize with **Docker** for reproducible deployment  
- Build a **FastAPI-based REST API** for remote inference  
- Extend training to more diverse PCB layouts and manufacturers  
- Add **real-time camera feed** integration for inline inspection systems  

---

##  14. Contributor

**Anumula Dinesh Reddy**  
Computer Science & Engineering  

---

##  15. License

This project is licensed under the **MIT License**.

---
