import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import io
import timm

# Page configuration
st.set_page_config(
    page_title="CircuitGuard - PCB Defect Detection",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Class labels
CLASS_NAMES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']

@st.cache_resource
def load_model(model_path):
    """Load the trained EfficientNet model"""
    try:
        # Create model architecture
        model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=6)
        
        # Try loading with different methods
        try:
            # Method 1: Try torch.load with weights_only=False
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
                
        except Exception as e1:
            st.warning(f"Torch load failed: {str(e1)}")
            try:
                # Method 2: Try loading with pickle directly
                import pickle
                with open(model_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
            except Exception as e2:
                st.error(f"All loading methods failed. Error: {str(e2)}")
                return None
        
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("💡 Try using the original .pth file instead of .pkl file")
        return None

def get_image_transforms():
    """Get image transformation pipeline"""
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def preprocess_image(image):
    """Preprocess uploaded image for model inference"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    transform = get_image_transforms()
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def predict_defect(model, image_tensor):
    """Run inference and get prediction"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    predicted_class = CLASS_NAMES[predicted.item()]
    confidence_score = confidence.item()
    
    # Get top 3 predictions
    top3_prob, top3_indices = torch.topk(probabilities, 3)
    top3_predictions = [
        (CLASS_NAMES[idx.item()], prob.item()) 
        for idx, prob in zip(top3_indices[0], top3_prob[0])
    ]
    
    return predicted_class, confidence_score, top3_predictions

def apply_image_subtraction(template_img, test_img):
    """Apply image subtraction to highlight defects"""
    # Convert to numpy arrays
    template_np = np.array(template_img)
    test_np = np.array(test_img)
    
    # Convert to grayscale
    template_gray = cv2.cvtColor(template_np, cv2.COLOR_RGB2GRAY)
    test_gray = cv2.cvtColor(test_np, cv2.COLOR_RGB2GRAY)
    
    # Resize to same dimensions
    h, w = min(template_gray.shape[0], test_gray.shape[0]), min(template_gray.shape[1], test_gray.shape[1])
    template_gray = cv2.resize(template_gray, (w, h))
    test_gray = cv2.resize(test_gray, (w, h))
    
    # Image subtraction
    diff = cv2.absdiff(template_gray, test_gray)
    
    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return diff, thresh

def detect_contours(thresh_image, original_image):
    """Detect contours and draw bounding boxes"""
    # Find contours
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Convert original to numpy for drawing
    result_image = np.array(original_image).copy()
    
    # Draw bounding boxes
    defect_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # Filter small noise
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            defect_regions.append((x, y, w, h))
    
    return result_image, defect_regions

def main():
    # Header
    st.markdown('<h1 class="main-header">🔌 CircuitGuard - PCB Defect Detection</h1>', unsafe_allow_html=True)
    st.markdown("### Automated defect detection and classification for Printed Circuit Boards")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model loading
        model_path = st.text_input(
            "Model Path",
            value="models/efficientnet_state_dict.pkl",
            help="Path to your trained model file (.pkl or .pth)"
        )
        
        if st.button("Load Model"):
            with st.spinner("Loading model..."):
                model = load_model(model_path)
                if model:
                    st.session_state['model'] = model
                    st.success("✅ Model loaded successfully!")
        
        st.divider()
        
        # Detection mode
        st.subheader("Detection Mode")
        detection_mode = st.radio(
            "Choose mode:",
            ["Classification Only", "Full Pipeline (with Template)"],
            help="Classification: Direct defect classification\nFull Pipeline: Image subtraction + classification"
        )
        
        st.divider()
        
        st.subheader("About")
        st.info("""
        **CircuitGuard** uses EfficientNet-B4 deep learning model to detect and classify PCB defects.
        
        **Defect Types:**
        - Missing Hole
        - Mouse Bite
        - Open Circuit
        - Short
        - Spur
        - Spurious Copper
        
        **Model Performance:** 97.76% accuracy
        """)
    
    # Main content
    if 'model' not in st.session_state:
        st.warning("⚠️ Please load the model from the sidebar first!")
        st.info("Enter the path to your `efficientnet_state_dict.pkl` file and click 'Load Model'")
        return
    
    model = st.session_state['model']
    
    # File uploaders based on mode
    if detection_mode == "Full Pipeline (with Template)":
        st.markdown('<h2 class="sub-header">📤 Upload Images</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Template Image (Defect-Free)")
            template_file = st.file_uploader(
                "Upload template PCB image",
                type=['png', 'jpg', 'jpeg'],
                key="template"
            )
            
        with col2:
            st.subheader("Test Image")
            test_file = st.file_uploader(
                "Upload test PCB image",
                type=['png', 'jpg', 'jpeg'],
                key="test"
            )
        
        if template_file and test_file:
            # Load images
            template_img = Image.open(template_file)
            test_img = Image.open(test_file)
            
            # Display original images
            st.markdown('<h2 class="sub-header">🖼️ Original Images</h2>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.image(template_img, caption="Template Image", use_container_width=True)
            with col2:
                st.image(test_img, caption="Test Image", use_container_width=True)
            
            if st.button("🔍 Detect Defects", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    # Image subtraction
                    diff, thresh = apply_image_subtraction(template_img, test_img)
                    
                    # Contour detection
                    result_img, defect_regions = detect_contours(thresh, test_img)
                    
                    # Display processing results
                    st.markdown('<h2 class="sub-header">🔬 Processing Results</h2>', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.image(diff, caption="Difference Map", use_container_width=True, clamp=True)
                    with col2:
                        st.image(thresh, caption="Thresholded Image", use_container_width=True, clamp=True)
                    with col3:
                        st.image(result_img, caption="Detected Defects", use_container_width=True)
                    
                    # Classification of defect regions
                    if len(defect_regions) > 0:
                        st.markdown('<h2 class="sub-header">🎯 Defect Classification</h2>', unsafe_allow_html=True)
                        st.write(f"**Found {len(defect_regions)} defect region(s)**")
                        
                        # Store results for export
                        predictions_list = []
                        
                        for idx, (x, y, w, h) in enumerate(defect_regions):
                            # Crop defect region
                            test_np = np.array(test_img)
                            defect_crop = test_np[y:y+h, x:x+w]
                            defect_pil = Image.fromarray(defect_crop)
                            
                            # Predict
                            img_tensor = preprocess_image(defect_pil)
                            pred_class, confidence, top3 = predict_defect(model, img_tensor)
                            
                            # Store prediction
                            predictions_list.append({
                                'region': f"Region_{idx+1}",
                                'class': pred_class,
                                'confidence': confidence,
                                'top3': top3,
                                'bbox': (x, y, w, h)
                            })
                            
                            # Display results
                            with st.expander(f"Defect Region {idx+1} - **{pred_class.upper()}** ({confidence*100:.2f}% confidence)"):
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    st.image(defect_pil, caption=f"Region {idx+1}", use_container_width=True)
                                with col2:
                                    st.write("**Top 3 Predictions:**")
                                    for i, (cls, prob) in enumerate(top3):
                                        st.metric(f"{i+1}. {cls}", f"{prob*100:.2f}%")
                        
                        # Store for export
                        st.session_state['last_result_image'] = result_img
                        st.session_state['last_predictions'] = predictions_list
                    else:
                        st.success("✅ No significant defects detected!")
                        # Clear previous results
                        st.session_state.pop('last_result_image', None)
                        st.session_state.pop('last_predictions', None)
    
    else:  # Classification Only mode
        st.markdown('<h2 class="sub-header">📤 Upload Image</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload PCB defect image",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔍 Classify Defect", type="primary", use_container_width=True):
                with st.spinner("Classifying..."):
                    # Preprocess and predict
                    img_tensor = preprocess_image(image)
                    pred_class, confidence, top3 = predict_defect(model, img_tensor)
                    
                    # Store for export
                    img_np = np.array(image)
                    st.session_state['last_result_image'] = img_np
                    st.session_state['last_predictions'] = [{
                        'region': 'Single_Image',
                        'class': pred_class,
                        'confidence': confidence,
                        'top3': top3,
                        'bbox': None
                    }]
                    
                    with col2:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.markdown(f"### 🎯 Prediction: **{pred_class.upper()}**")
                        st.markdown(f"### 📊 Confidence: **{confidence*100:.2f}%**")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Confidence indicator
                        if confidence > 0.9:
                            st.success("High confidence prediction")
                        elif confidence > 0.7:
                            st.info("Moderate confidence prediction")
                        else:
                            st.warning("Low confidence - manual review recommended")
                    
                    # Display top 3 predictions
                    st.markdown('<h2 class="sub-header">📈 Top 3 Predictions</h2>', unsafe_allow_html=True)
                    cols = st.columns(3)
                    for i, (cls, prob) in enumerate(top3):
                        with cols[i]:
                            st.metric(
                                label=f"#{i+1}: {cls}",
                                value=f"{prob*100:.2f}%"
                            )
    
    # Download section
    st.divider()
    st.markdown('<h2 class="sub-header">💾 Export Results</h2>', unsafe_allow_html=True)
    
    # Check if we have results to export
    if 'last_result_image' in st.session_state or 'last_predictions' in st.session_state:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'last_result_image' in st.session_state:
                # Convert numpy array to PIL Image
                result_img = Image.fromarray(st.session_state['last_result_image'])
                
                # Convert to bytes
                buf = io.BytesIO()
                result_img.save(buf, format='PNG')
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="📥 Download Annotated Image",
                    data=byte_im,
                    file_name="circuitguard_annotated_result.png",
                    mime="image/png",
                    use_container_width=True
                )
        
        with col2:
            if 'last_predictions' in st.session_state:
                import csv
                from datetime import datetime
                
                # Create CSV content
                csv_buffer = io.StringIO()
                writer = csv.writer(csv_buffer)
                
                # Write header
                writer.writerow(['Timestamp', 'Defect_Region', 'Predicted_Class', 'Confidence', 'Top_2', 'Top_3'])
                
                # Write predictions
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for pred in st.session_state['last_predictions']:
                    writer.writerow([
                        timestamp,
                        pred['region'],
                        pred['class'],
                        f"{pred['confidence']:.4f}",
                        f"{pred['top3'][1][0]}: {pred['top3'][1][1]:.4f}",
                        f"{pred['top3'][2][0]}: {pred['top3'][2][1]:.4f}"
                    ])
                
                st.download_button(
                    label="📄 Download Prediction Log",
                    data=csv_buffer.getvalue(),
                    file_name="circuitguard_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    else:
        st.info("💡 Run a prediction first to enable export options")

if __name__ == "__main__":
    main()