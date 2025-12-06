🚀 CircuitGuard — PCB Defect Detection & Classification System
CircuitGuard is an AI-powered system that detects, localizes, and classifies defects in Printed 
Circuit Boards (PCBs) using a hybrid pipeline of image subtraction, contour-based ROI
extraction, and a deep learning model trained with EfficientNet-B4.
The system also includes a Streamlit web application that allows users to upload PCB images and instantly see annotated defect predictions.

✨ Key Features
🔍 Automated PCB Defect Detection using template–test subtraction

📦 ROI Segmentation with OpenCV contour extraction

🤖 Defect Classification using EfficientNet-B4 (PyTorch)

🌐 Streamlit Web App for uploads, predictions, and visualization

📥 Exports Annotated Images & Prediction Logs

📊 Model Evaluation Tools (accuracy, loss, confusion matrix)

🧠 System Workflow
Template Image + Test Image
            │
      Image Subtraction
            │
      Otsu Thresholding
            │
  Erosion + Dilation Filters
            │
      Contour Extraction
            │
   ROI Segmentation (Cropped)
            │
 EfficientNet-B4 Classification
            │
  Annotated Output (Web UI)

🛠️ Tech Stack
Area	Tools
Image Processing	OpenCV, NumPy
Model	PyTorch, EfficientNet-B4 (timm)
UI	Streamlit
Backend	Python
Export	CSV, Annotated Images

🚀 Setup & Installation
1. Clone the Repository

     git clone https://github.com/username/CircuitGuard.git
   
     cd CircuitGuard

2. Install Dependencies
   
pip install -r requirements.txt

3. Run Preprocessing Scripts
   
      python preprocessing/subtraction.py
   
      python roi_extraction/contour_detect.py

4. Train the Model
   
     python model/train.py

5. Launch the Streamlit App
   
   streamlit run app.py

🧪 Model Performance

✔ EfficientNet-B4 achieving ≥97% accuracy 

✔ Confusion matrix and training curves exported after training

✔ Robust prediction consistency on unseen test images

🌐 Streamlit Web Application

The UI provides:

📤 Upload fields for template and test images

🧠 Automatic processing through backend pipeline

🟩 Annotated output images with bounding boxes & defect labels

📥 Button to download annotated results + logs
