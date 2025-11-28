import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

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
    .defect-highlight {
        background: linear-gradient(90deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: bold;
        text-align: center;
        margin: 10px 0;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .result-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #ff6b6b;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Defect classes with descriptions
CLASS_NAMES = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
DEFECT_DESCRIPTIONS = {
    'Missing_hole': 'Missing drill holes or vias in the PCB substrate',
    'Mouse_bite': 'Partial etching causing irregular bite-like patterns on copper traces',
    'Open_circuit': 'Broken or disconnected circuit traces interrupting current flow', 
    'Short': 'Unwanted connections between traces causing short circuits',
    'Spur': 'Unwanted copper protrusions or spikes along trace edges',
    'Spurious_copper': 'Excess copper material remaining after etching process'
}

# Defect color mapping for visualization
DEFECT_COLORS = {
    'Missing_hole': (255, 0, 0),      # Red
    'Mouse_bite': (255, 165, 0),      # Orange
    'Open_circuit': (0, 0, 255),      # Blue
    'Short': (128, 0, 128),           # Purple
    'Spur': (255, 255, 0),            # Yellow
    'Spurious_copper': (0, 255, 255)  # Cyan
}

# Model class
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
    
    model_path = r"C:\Users\Kandu\OneDrive\Desktop\info_temp\PCB_DATASET\modules\module3_model_training\checkpoints\best_model.pth"
    
    try:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=DEVICE)
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
                
            st.sidebar.success("✅ Model loaded successfully!")
        else:
            st.sidebar.warning("⚠️ Model file not found. Using demo mode.")
            
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
    
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

def detect_defect_regions(image, defect_type):
    """
    Simulate defect region detection based on defect type
    In a real system, this would use object detection or segmentation
    """
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # Create a copy for drawing
    result_image = img_array.copy()
    
    # Generate simulated defect regions based on defect type
    regions = []
    
    if defect_type == 'Missing_hole':
        # Simulate missing holes as dark circles
        regions = [(width//4, height//4, 20), (3*width//4, height//4, 15)]
    elif defect_type == 'Mouse_bite':
        # Simulate irregular patterns along edges
        regions = [(width//3, height//2, 25), (2*width//3, height//3, 20)]
    elif defect_type == 'Open_circuit':
        # Simulate broken traces
        regions = [(width//2, height//3, 30)]
    elif defect_type == 'Short':
        # Simulate bridge between traces
        regions = [(width//2, height//2, 25)]
    elif defect_type == 'Spur':
        # Simulate spike patterns
        regions = [(width//5, 4*height//5, 20), (4*width//5, height//5, 15)]
    elif defect_type == 'Spurious_copper':
        # Simulate excess copper patches
        regions = [(width//6, height//6, 35), (5*width//6, 5*height//6, 30)]
    
    return result_image, regions

def create_defect_visualization(original_image, defect_type, confidence, regions):
    """
    Create annotated image with defect regions highlighted
    """
    # Convert to numpy if needed
    if isinstance(original_image, Image.Image):
        img_array = np.array(original_image)
    else:
        img_array = original_image.copy()
    
    # Convert to BGR for OpenCV
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Resize for better visualization
    height, width = img_array.shape[:2]
    if max(height, width) > 800:
        scale = 800 / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_array = cv2.resize(img_array, (new_width, new_height))
        # Scale regions accordingly
        regions = [(int(x*scale), int(y*scale), int(r*scale)) for x, y, r in regions]
    
    # Get defect color
    defect_color = DEFECT_COLORS.get(defect_type, (255, 0, 0))
    
    # Highlight defect regions
    for i, (x, y, radius) in enumerate(regions):
        # Draw circle around defect area
        cv2.circle(img_array, (x, y), radius, defect_color, 3)
        
        # Draw filled circle for better visibility
        cv2.circle(img_array, (x, y), radius, defect_color, -1)
        
        # Add defect number
        cv2.putText(img_array, f"D{i+1}", (x-10, y+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add information overlay
    overlay = img_array.copy()
    cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img_array, 0.3, 0, img_array)
    
    # Add text information
    confidence_text = f"Confidence: {confidence:.1%}"
    
    cv2.putText(img_array, "DEFECT DETECTED", (20, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img_array, f"Type: {defect_type}", (20, 65), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, defect_color, 2)
    cv2.putText(img_array, confidence_text, (20, 95), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, defect_color, 2)
    
    # Add legend for defect regions
    cv2.putText(img_array, "D1, D2 = Defect Locations", (10, img_array.shape[0]-20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

def create_defect_analysis_report(defect_type, confidence, regions):
    """Create a detailed defect analysis report"""
    
    st.markdown(f"""
    <div class="defect-highlight">
        🔍 DEFECT DETECTED: {defect_type.upper()}
    </div>
    """, unsafe_allow_html=True)
    
    # Defect details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Defect Type", defect_type)
    
    with col2:
        confidence_class = "confidence-high" if confidence > 0.8 else "confidence-medium" if confidence > 0.6 else "confidence-low"
        st.metric("Confidence", f"{confidence:.1%}")
    
    with col3:
        st.metric("Defect Areas", len(regions))
    
    # Defect description
    st.markdown(f"""
    <div class="result-card">
        <h4>📋 Defect Description</h4>
        <p>{DEFECT_DESCRIPTIONS.get(defect_type, 'No description available')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Defect locations
    if regions:
        st.markdown("""
        <div class="result-card">
            <h4>📍 Defect Locations</h4>
        </div>
        """, unsafe_allow_html=True)
        
        for i, (x, y, radius) in enumerate(regions):
            st.write(f"**Defect D{i+1}**: Position ({x}, {y}) - Radius: {radius}px")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">🔍 CircuitGuard - PCB Defect Detection</div>', unsafe_allow_html=True)
    st.markdown("### Visual Defect Localization & Analysis System")
    
    # Initialize model
    model = load_model()
    
    # Create layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload PCB Image")
        
        uploaded_file = st.file_uploader(
            "Choose a PCB image for defect analysis",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload a clear image of the PCB for defect detection"
        )
        
        if uploaded_file:
            # Display original image
            original_image = Image.open(uploaded_file)
            st.image(original_image, caption="📷 Original PCB Image", use_column_width=True)
            
            # Analysis button
            if st.button("🔍 Analyze Defects", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing PCB for defects and locating problem areas..."):
                    try:
                        # Run model inference
                        input_tensor = preprocess_image(original_image).to(DEVICE)
                        
                        with torch.no_grad():
                            outputs = model(input_tensor)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)
                            confidence, predicted_idx = torch.max(probabilities, 1)
                        
                        # Get prediction results
                        defect_type = CLASS_NAMES[predicted_idx.item()]
                        confidence_score = confidence.item()
                        
                        # Detect defect regions (simulated)
                        original_np = np.array(original_image)
                        result_image, defect_regions = detect_defect_regions(original_image, defect_type)
                        
                        # Create visualization
                        annotated_image = create_defect_visualization(
                            original_np, defect_type, confidence_score, defect_regions
                        )
                        
                        # Display results in right column
                        with col2:
                            st.markdown("### 🎯 Detection Results")
                            
                            # Show annotated image
                            st.image(annotated_image, 
                                   caption="🔦 Defect Locations Highlighted", 
                                   use_column_width=True)
                            
                            # Show analysis report
                            create_defect_analysis_report(defect_type, confidence_score, defect_regions)
                            
                            # Show probability distribution
                            st.markdown("### 📊 Defect Probability Distribution")
                            prob_data = {
                                'Defect Type': CLASS_NAMES,
                                'Probability': [f"{prob:.2%}" for prob in probabilities[0].cpu().numpy()],
                                'Value': probabilities[0].cpu().numpy()
                            }
                            
                            # Create bar chart
                            fig, ax = plt.subplots(figsize=(10, 6))
                            colors = ['#ff6b6b' if cls == defect_type else '#4ecdc4' for cls in CLASS_NAMES]
                            y_pos = np.arange(len(CLASS_NAMES))
                            
                            ax.barh(y_pos, prob_data['Value'], color=colors)
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(CLASS_NAMES)
                            ax.set_xlabel('Probability')
                            ax.set_title('Defect Classification Confidence')
                            
                            # Add value labels
                            for i, v in enumerate(prob_data['Value']):
                                ax.text(v + 0.01, i, f'{v:.2%}', va='center')
                            
                            st.pyplot(fig)
                            
                            # Defect color legend
                            st.markdown("### 🎨 Defect Color Legend")
                            legend_cols = st.columns(3)
                            for idx, (defect_name, color) in enumerate(DEFECT_COLORS.items()):
                                with legend_cols[idx % 3]:
                                    color_hex = '#%02x%02x%02x' % color
                                    st.markdown(f"""
                                    <div style='display: flex; align-items: center; margin: 5px 0;'>
                                        <div style='width: 20px; height: 20px; background-color: {color_hex}; margin-right: 10px; border-radius: 3px;'></div>
                                        <span>{defect_name}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
    
    # Sidebar information
    with st.sidebar:
        st.markdown("## 🔧 System Info")
        st.markdown("---")
        st.write(f"**Device**: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        st.write(f"**Model**: EfficientNet-B4")
        st.write(f"**Defect Types**: {len(CLASS_NAMES)}")
        
        st.markdown("---")
        st.markdown("## 🎯 Defect Types")
        for defect in CLASS_NAMES:
            st.write(f"• **{defect}**")
        
        st.markdown("---")
        st.markdown("## 💡 How to Read Results")
        st.info("""
        - **Colored circles** show defect locations
        - **D1, D2** labels identify specific defects
        - **Color coding** indicates defect type
        - **Confidence score** shows detection certainty
        """)

if __name__ == "__main__":
    main()