import streamlit as st
import torch
from PIL import Image
import numpy as np
import os
import time
import sys
import matplotlib.pyplot as plt

# Add backend to path
sys.path.append(os.path.dirname(__file__))
from backend_engine import PCBDefectDetector, DetectionResult

# Set page configuration
st.set_page_config(
    page_title="CircuitGuard - PCB Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for optimized performance
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
    .performance-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 5px;
    }
    .defect-highlight {
        background: linear-gradient(90deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: bold;
        text-align: center;
        margin: 10px 0;
    }
    .result-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #ff6b6b;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class PCBDefectApp:
    """Streamlit application with optimized backend integration"""
    
    def __init__(self):
        self.detector = None
        self.model_path = r"C:\Users\Kandu\OneDrive\Desktop\info_temp\PCB_DATASET\modules\module3_model_training\checkpoints\best_model.pth"
        self.initialize_backend()
    
    def initialize_backend(self):
        """Initialize the backend detection engine"""
        try:
            if 'detector' not in st.session_state:
                st.session_state.detector = PCBDefectDetector(self.model_path)
                st.session_state.backend_ready = True
                st.sidebar.success("✅ Backend engine initialized!")
            
            self.detector = st.session_state.detector
            
        except Exception as e:
            st.sidebar.error(f"❌ Backend initialization failed: {e}")
            st.session_state.backend_ready = False
    
    def display_performance_metrics(self, processing_time: float, image_size: tuple):
        """Display performance metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="performance-metric">
                <h4>⏱️ Processing Time</h4>
                <h3>{processing_time:.2f}s</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            device = "GPU" if torch.cuda.is_available() else "CPU"
            st.markdown(f"""
            <div class="performance-metric">
                <h4>⚡ Device</h4>
                <h3>{device}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="performance-metric">
                <h4>🖼️ Image Size</h4>
                <h3>{image_size[0]}x{image_size[1]}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            fps = 1.0 / processing_time if processing_time > 0 else 0
            st.markdown(f"""
            <div class="performance-metric">
                <h4>📊 Throughput</h4>
                <h3>{fps:.1f} FPS</h3>
            </div>
            """, unsafe_allow_html=True)
    
    def display_detection_results(self, result: DetectionResult):
        """Display detection results with optimized layout"""
        
        # Defect highlight
        st.markdown(f"""
        <div class="defect-highlight">
            🔍 DEFECT DETECTED: {result.defect_type.upper()}
        </div>
        """, unsafe_allow_html=True)
        
        # Performance metrics
        self.display_performance_metrics(result.processing_time, result.original_size)
        
        # Results in columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Annotated image
            st.image(result.annotated_image, 
                   caption="🔦 Defect Locations Highlighted", 
                   use_column_width=True)
            
            # Defect regions info
            if result.defect_regions:
                st.markdown("""
                <div class="result-card">
                    <h4>📍 Defect Locations</h4>
                </div>
                """, unsafe_allow_html=True)
                
                for i, (x, y, radius) in enumerate(result.defect_regions):
                    st.write(f"**Defect D{i+1}**: Position ({x}, {y}) - Radius: {radius}px")
        
        with col2:
            # Confidence and details
            confidence_col1, confidence_col2 = st.columns(2)
            
            with confidence_col1:
                st.metric("Defect Type", result.defect_type)
            
            with confidence_col2:
                confidence_class = "high" if result.confidence > 0.8 else "medium" if result.confidence > 0.6 else "low"
                st.metric("Confidence", f"{result.confidence:.1%}")
            
            # Probability distribution
            st.markdown("### 📊 Defect Probability Distribution")
            
            # Create optimized chart
            fig, ax = plt.subplots(figsize=(8, 4))
            classes = list(result.all_probabilities.keys())
            probabilities = list(result.all_probabilities.values())
            colors = ['#ff6b6b' if cls == result.defect_type else '#4ecdc4' for cls in classes]
            
            bars = ax.barh(classes, probabilities, color=colors)
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability')
            ax.set_title('Defect Classification Confidence')
            
            # Add value labels
            for bar, prob in zip(bars, probabilities):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{prob:.2%}', ha='left', va='center', fontsize=9)
            
            st.pyplot(fig, use_container_width=True)
    
    def process_uploaded_image(self, uploaded_file):
        """Process uploaded image with backend engine"""
        if not st.session_state.backend_ready:
            st.error("Backend engine not ready. Please check model file.")
            return
        
        try:
            # Display original image
            original_image = Image.open(uploaded_file)
            st.image(original_image, caption="📷 Original PCB Image", use_column_width=True)
            
            # Process with backend
            with st.spinner("🔄 Analyzing PCB with optimized backend..."):
                result = self.detector.process_image(original_image)
            
            # Display results
            self.display_detection_results(result)
            
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
    
    def run(self):
        """Run the main application"""
        # Header
        st.markdown('<div class="main-header">🔍 CircuitGuard - PCB Defect Detection</div>', unsafe_allow_html=True)
        st.markdown("### Optimized Backend Pipeline with Real-time Processing")
        
        # Check backend status
        if not st.session_state.backend_ready:
            st.warning("⚠️ Backend engine not initialized. Check if model file exists.")
            return
        
        # Main layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📤 Upload PCB Image")
            
            uploaded_file = st.file_uploader(
                "Choose a PCB image for defect analysis",
                type=['png', 'jpg', 'jpeg', 'bmp'],
                help="Upload a clear image of the PCB for defect detection"
            )
            
            if uploaded_file:
                self.process_uploaded_image(uploaded_file)
        
        with col2:
            if 'last_result' not in st.session_state:
                st.markdown("### 🎯 Detection Results")
                st.info("👆 Upload an image to see defect analysis results here")
        
        # Sidebar with system info
        self.display_sidebar_info()
    
    def display_sidebar_info(self):
        """Display system information in sidebar"""
        with st.sidebar:
            st.markdown("## 🔧 System Information")
            st.markdown("---")
            
            if self.detector:
                system_info = self.detector.get_system_info()
                st.write(f"**Device**: {system_info['device']}")
                st.write(f"**Backend Status**: {'✅ Ready' if system_info['model_initialized'] else '❌ Not Ready'}")
                st.write(f"**CUDA Available**: {'✅ Yes' if system_info['cuda_available'] else '❌ No'}")
            
            st.markdown("---")
            st.markdown("## 🚀 Performance Tips")
            st.info("""
            - Use **GPU** for faster processing
            - Keep image size **under 2000px** for optimal speed
            - **JPG format** loads faster than PNG
            - Close other applications for better performance
            """)
            
            st.markdown("---")
            st.markdown("## 📊 Backend Features")
            st.success("""
            ✅ Modular backend engine
            ✅ Optimized preprocessing  
            ✅ Batch processing ready
            ✅ Real-time performance
            ✅ Comprehensive logging
            ✅ Error handling
            """)

def main():
    """Main application entry point"""
    app = PCBDefectApp()
    app.run()

if __name__ == "__main__":
    main()