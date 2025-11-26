# CircuitGuard: AI-Powered PCB Defect Detection 🛡️

**Automated. Accurate. Aesthetic.**

CircuitGuard is a state-of-the-art Printed Circuit Board (PCB) inspection system designed to detect manufacturing defects with high precision. It features a dual-mode architecture, combining traditional reference-based inspection with cutting-edge YOLOv8 AI for single-image detection, all wrapped in a stunning, premium dark-mode interface.

## 🌟 Key Features

*   **🚀 Dual-Mode Detection**:
    *   **Reference Mode**: Compares a test image against a "Golden Template" to find subtle deviations using structural similarity and EfficientNet.
    *   **AI Mode (YOLOv8)**: Instantly detects defects in a single image using a custom-trained YOLOv8 Nano model. No template required!
*   **🎨 Premium Glassmorphism UI**: A fully responsive, modern frontend featuring a deep indigo dark theme, realistic glass effects, and smooth micro-interactions.
*   **⚡ Real-Time Inference**: Optimized backend pipeline ensuring sub-second analysis.
*   **📊 Comprehensive Reporting**: Visualizes defects directly on the PCB image and exports detailed CSV logs for quality control.

## 🛠️ Tech Stack

*   **AI Core**: YOLOv8 (Ultralytics), PyTorch, OpenCV
*   **Backend**: FastAPI (Python)
*   **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism), JavaScript
*   **Processing**: NumPy, Pillow

## 🚀 Getting Started

Follow these steps to set up CircuitGuard on your local machine.

### Prerequisites
*   Python 3.8+
*   Git

### 1. Clone the Repository
```bash
git clone -b jay_zawar https://github.com/mentor-pranaya/CircuitGuard-PCB-Defect-Detection-and-Classification-Application_1.git
cd CircuitGuard-PCB-Defect-Detection-and-Classification-Application_1
```

### 2. Install Dependencies
```bash
pip install ultralytics fastapi uvicorn opencv-python-headless pillow numpy torch torchvision torchaudio
```

### 3. Run the Backend Server
Start the FastAPI server to handle AI inference:
```bash
python backend.py
```
*The server will start at `http://localhost:8000`.*

### 4. Launch the Application
Simply open the `frontend/index.html` file in your web browser.
*   **Tip**: For the best experience, use a modern browser like Chrome or Edge.

## 📖 How to Use

1.  **Select Mode**: Toggle between **Reference Mode** (requires template) and **AI Mode** (single image).
2.  **Upload Images**: Drag and drop your PCB images into the glass cards.
3.  **Analyze**: Click "Analyze Circuit" and watch the AI work its magic.
4.  **Review & Export**: Inspect the detected defects on the canvas and download the report.

## 📂 Project Structure

*   `backend.py`: The heart of the application. Handles API requests and orchestrates AI models.
*   `best_yolo.pt`: Our custom-trained YOLOv8 model weights.
*   `frontend/`: Contains the beautiful UI code (`index.html`, `styles.css`, `script.js`).
*   `prepare_yolo_data.py`: Utility script used to convert XML annotations to YOLO format.
*   `train_yolo.py`: Script used to train the YOLOv8 model.

---
*Developed by Jay Zawar*
