import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import cv2
import numpy as np
import os
import time

# Set page configuration
st.set_page_config(
    page_title="CircuitGuard - PCB Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.4rem;
        color: #2ca02c;
        margin-bottom: 1rem;
        font-weight: bold;
        border-left: 4px solid #2ca02c;
        padding-left: 10px;
    }
    .upload-section {
        background: rgba(255, 255, 255, 0.95);
        padding: 25px;
        border-radius: 15px;
        border: 2px dashed #1f77b4;
        margin: 20px 0;
        text-align: center;
    }
    .result-section {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #28a745;
        margin: 15px 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .defect-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Defect classes
CLASS_NAMES = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
CLASS_DESCRIPTIONS = {
    'Missing_hole': 'Missing drill holes or vias in PCB',
    'Mouse_bite': 'Partial etching causing irregular patterns',
    'Open_circuit': 'Broken or disconnected circuit traces',
    'Short': 'Unwanted connections between traces',
    'Spur': 'Unwanted copper protrusions or spikes',
    'Spurious_copper': 'Excess copper material remaining'
}

# Model class (same as training)
class EfficientNetB4Classifier(nn.Module):
    def __init__(self, num_classes=6):
        super(EfficientNetB4Classifier, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

@st.cache_resource
def load_model():
    """Load the trained PCB defect detection model"""
    model = EfficientNetB4Classifier(num_classes=len(CLASS_NAMES))
    
    model_path = r"C:\Users\Kandu\OneDrive\Desktop\PCB_DATASET\best_model.pth"
    
    try:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=DEVICE)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    # Assume it's the state dict itself
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
                
            st.success("✅ **Model loaded successfully!**")
        else:
            st.warning("⚠️ **Model file not found. Using pretrained weights for demonstration.**")
            
    except Exception as e:
        st.error(f"❌ **Error loading model: {e}**")
        st.info("Using pretrained weights for demonstration.")
    
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(image):
    """Preprocess image for model inference"""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    return transform(image).unsqueeze(0)

def create_annotated_image(image, prediction, original_size=None):
    """Create annotated image with bounding boxes and labels"""
    if isinstance(image, np.ndarray):
        annotated_image = image.copy()
    else:
        annotated_image = np.array(image)
    
    # Convert to BGR for OpenCV
    if len(annotated_image.shape) == 3 and annotated_image.shape[2] == 3:
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    
    # Resize for better visualization if needed
    h, w = annotated_image.shape[:2]
    if max(h, w) > 800:
        scale = 800 / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        annotated_image = cv2.resize(annotated_image, new_size)
    
    # Add prediction information
    defect_class = prediction['class']
    confidence = prediction['confidence']
    
    # Choose color based on confidence
    if confidence > 0.8:
        color = (0, 255, 0)  # Green
        status = "HIGH CONFIDENCE"
    elif confidence > 0.6:
        color = (0, 165, 255)  # Orange
        status = "MEDIUM CONFIDENCE"
    else:
        color = (0, 0, 255)  # Red
        status = "LOW CONFIDENCE"
    
    # Add semi-transparent overlay for text
    overlay = annotated_image.copy()
    cv2.rectangle(overlay, (10, 10), (500, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, annotated_image, 0.3, 0, annotated_image)
    
    # Add text information
    cv2.putText(annotated_image, "PCB DEFECT DETECTION", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(annotated_image, f"Defect: {defect_class}", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(annotated_image, f"Confidence: {confidence:.2%}", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(annotated_image, f"Status: {status}", (20, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Add bounding box around the image
    cv2.rectangle(annotated_image, (5, 5), (annotated_image.shape[1]-5, annotated_image.shape[0]-5), color, 3)
    
    return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">🔍 CircuitGuard - PCB Defect Detection</div>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Automated PCB Quality Control System")
    
    # Initialize model
    model = load_model()
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="sub-header">📤 Image Upload</div>', unsafe_allow_html=True)
        
        # Upload section
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        
        upload_option = st.radio(
            "Select input method:",
            ["Upload Template Image", "Upload Test Image", "Both Template and Test"],
            horizontal=True
        )
        
        uploaded_files = []
        
        if upload_option in ["Upload Template Image", "Both Template and Test"]:
            template_file = st.file_uploader(
                "Upload Template PCB Image (Reference)",
                type=['png', 'jpg', 'jpeg', 'bmp'],
                key="template"
            )
            if template_file:
                uploaded_files.append(("Template", template_file))
        
        if upload_option in ["Upload Test Image", "Both Template and Test"]:
            test_file = st.file_uploader(
                "Upload Test PCB Image (For Analysis)",
                type=['png', 'jpg', 'jpeg', 'bmp'],
                key="test"
            )
            if test_file:
                uploaded_files.append(("Test", test_file))
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display uploaded images
        if uploaded_files:
            st.markdown('<div class="sub-header">🖼️ Uploaded Images</div>', unsafe_allow_html=True)
            
            for img_type, uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"{img_type} Image: {uploaded_file.name}", use_column_width=True)
    
    with col2:
        st.markdown('<div class="sub-header">🔍 Analysis Results</div>', unsafe_allow_html=True)
        
        # Analysis button
        if uploaded_files and st.button("🚀 Start Defect Analysis", type="primary", use_container_width=True):
            
            with st.spinner("🔍 Analyzing PCB images for defects..."):
                progress_bar = st.progress(0)
                
                for i, (img_type, uploaded_file) in enumerate(uploaded_files):
                    progress_bar.progress((i + 1) * 50)
                    
                    try:
                        # Load and preprocess image
                        image = Image.open(uploaded_file)
                        image_np = np.array(image)
                        
                        # Run inference
                        input_tensor = preprocess_image(image).to(DEVICE)
                        
                        with torch.no_grad():
                            outputs = model(input_tensor)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)
                            confidence, predicted_idx = torch.max(probabilities, 1)
                        
                        # Get prediction results
                        prediction = {
                            'class': CLASS_NAMES[predicted_idx.item()],
                            'confidence': confidence.item(),
                            'all_probabilities': probabilities.cpu().numpy()[0]
                        }
                        
                        # Create annotated image
                        annotated_image = create_annotated_image(image_np, prediction)
                        
                        # Display results
                        st.markdown('<div class="result-section">', unsafe_allow_html=True)
                        
                        # Defect type with confidence
                        confidence_class = "confidence-high" if prediction['confidence'] > 0.8 else "confidence-medium" if prediction['confidence'] > 0.6 else "confidence-low"
                        
                        st.markdown(f"""
                        <div class="defect-card">
                            <h3>🎯 {img_type} Image Analysis</h3>
                            <p><strong>Defect Type:</strong> {prediction['class']}</p>
                            <p><strong>Description:</strong> {CLASS_DESCRIPTIONS.get(prediction['class'], 'No description available')}</p>
                            <p class="{confidence_class}">Confidence: {prediction['confidence']:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display annotated image
                        st.image(annotated_image, caption=f"Annotated {img_type} Image", use_column_width=True)
                        
                        # Probability distribution
                        st.subheader("📊 Defect Probability Distribution")
                        
                        prob_data = {
                            'Defect Type': CLASS_NAMES,
                            'Probability': [f"{prob:.2%}" for prob in prediction['all_probabilities']],
                            'Value': prediction['all_probabilities']
                        }
                        
                        # Display probabilities
                        for j, (cls, prob, val) in enumerate(zip(prob_data['Defect Type'], 
                                                               prob_data['Probability'], 
                                                               prob_data['Value'])):
                            if j == predicted_idx.item():
                                st.markdown(f"**🎯 {cls}**: {prob} ✅")
                            else:
                                st.markdown(f"{cls}: {prob}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error processing {img_type} image: {e}")
                
                progress_bar.progress(100)
                st.success("✅ Analysis completed successfully!")
        
        else:
            # Placeholder for results
            st.markdown("""
            <div style='text-align: center; padding: 40px; background: #f8f9fa; border-radius: 10px;'>
                <h3 style='color: #6c757d;'>👆 Upload images and click 'Start Defect Analysis'</h3>
                <p style='color: #6c757d;'>Results will appear here after analysis</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Sidebar with system information
    with st.sidebar:
        st.markdown("## 🔧 System Information")
        st.markdown("---")
        
        # System metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Device", "GPU" if torch.cuda.is_available() else "CPU")
        with col2:
            st.metric("Model", "EfficientNet-B4")
        
        st.markdown("---")
        st.markdown("### 🎯 Defect Classes")
        for i, cls in enumerate(CLASS_NAMES):
            st.write(f"{i+1}. **{cls}**")
            st.caption(CLASS_DESCRIPTIONS.get(cls, ""))
        
        st.markdown("---")
        st.markdown("### 📊 Model Performance")
        st.info("""
        - **Accuracy**: >91% on test data
        - **Input Size**: 128×128 pixels
        - **Classes**: 6 defect types
        - **Framework**: PyTorch
        """)
        
        st.markdown("---")
        st.markdown("### 💡 How to Use")
        st.info("""
        1. Upload PCB images (template/test/both)
        2. Click 'Start Defect Analysis'
        3. View defect classification results
        4. Check confidence scores
        5. Review annotated images
        """)

if __name__ == "__main__":
    main()