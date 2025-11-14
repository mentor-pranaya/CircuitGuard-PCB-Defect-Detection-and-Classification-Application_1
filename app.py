import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import tempfile
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="CircuitGuard - PCB Defect Detection",
    page_icon="🔍",
    layout="wide"
)


def main():
    # Header
    st.title("🔍 CircuitGuard - PCB Defect Detection System")
    st.markdown("""
    **Automated PCB defect detection and classification using computer vision and deep learning**
    """)

    st.markdown("---")

    # File upload section
    st.header("📤 Upload PCB Images")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Template Image")
        st.markdown("*Upload a reference image of a **good PCB***")
        template_file = st.file_uploader(
            "Choose template image",
            type=['jpg', 'jpeg', 'png'],
            key="template"
        )

        if template_file:
            template_image = Image.open(template_file)
            st.image(template_image, caption="Template Image",
                     use_column_width=True)

    with col2:
        st.subheader("Test Image")
        st.markdown("*Upload the PCB image you want to **check for defects***")
        test_file = st.file_uploader(
            "Choose test image",
            type=['jpg', 'jpeg', 'png'],
            key="test"
        )

        if test_file:
            test_image = Image.open(test_file)
            st.image(test_image, caption="Test Image", use_column_width=True)

    # Process button
    st.markdown("---")
    if st.button("🚀 Analyze PCB for Defects", type="primary", use_container_width=True):
        if template_file and test_file:
            analyze_defects(template_file, test_file)
        else:
            st.error("❌ Please upload both template and test images")

    # Information section
    st.markdown("---")
    st.header("📋 Supported Defect Types")

    defect_types = {
        "Missing Hole": "Holes that are absent in the test PCB",
        "Mouse Bite": "Partial holes or incomplete drilling",
        "Open Circuit": "Breaks in circuit traces",
        "Short Circuit": "Unintended connections between traces",
        "Spur": "Unwanted copper projections",
        "Spurious Copper": "Excess copper material"
    }

    for defect, description in defect_types.items():
        st.markdown(f"**• {defect}**: {description}")


def analyze_defects(template_file, test_file):
    """Analyze PCB images for defects"""

    with st.spinner("🔍 Analyzing PCB images for defects..."):

        # Convert to OpenCV format
        template_img = Image.open(template_file)
        test_img = Image.open(test_file)

        template_cv = cv2.cvtColor(np.array(template_img), cv2.COLOR_RGB2BGR)
        test_cv = cv2.cvtColor(np.array(test_img), cv2.COLOR_RGB2BGR)

        # Display processing steps
        st.header("📊 Analysis Results")

        # Create columns for results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🔬 Processing Steps")
            st.markdown("""
            1. **Image Alignment** - Align test image with template
            2. **Subtraction** - Find differences between images  
            3. **Thresholding** - Convert to binary mask
            4. **Contour Detection** - Find defect boundaries
            5. **Classification** - Identify defect types using AI
            """)

        with col2:
            st.subheader("🎯 Defects Found")

            # Sample defect results (you'll replace with actual detection)
            defect_results = [
                {"type": "Missing Hole", "count": 2, "confidence": "96%"},
                {"type": "Open Circuit", "count": 1, "confidence": "92%"},
                {"type": "Spur", "count": 1, "confidence": "88%"}
            ]

            for defect in defect_results:
                st.markdown(f"""
                **{defect['type']}**
                - Count: {defect['count']}
                - Confidence: {defect['confidence']}
                """)

        with col3:
            st.subheader("📈 Quality Metrics")
            st.metric("Total Defects", "4")
            st.metric("Detection Confidence", "94%")
            st.metric("Processing Time", "2.3s")

        # Visualization
        st.markdown("---")
        st.header("🖼️ Defect Visualization")

        # Create sample visualization (replace with actual results)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Template image
        ax1.imshow(template_img)
        ax1.set_title("Template PCB")
        ax1.axis('off')

        # Test image with defects
        ax2.imshow(test_img)
        ax2.set_title("Test PCB with Defects")
        ax2.axis('off')

        # Defect mask (sample)
        defect_mask = np.random.rand(100, 100)
        ax3.imshow(defect_mask, cmap='hot')
        ax3.set_title("Defect Heatmap")
        ax3.axis('off')

        st.pyplot(fig)

        # Download section
        st.markdown("---")
        st.header("📥 Download Results")

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="💾 Download Defect Report",
                data=generate_report(defect_results),
                file_name="circuitguard_defect_report.txt",
                mime="text/plain"
            )

        with col2:
            st.download_button(
                label="🖼️ Download Annotated Image",
                data=generate_sample_image(),
                file_name="defect_analysis.jpg",
                mime="image/jpeg"
            )


def generate_report(defect_results):
    """Generate a text report of defects found"""
    report = "CircuitGuard PCB Defect Analysis Report\n"
    report += "=" * 50 + "\n\n"

    report += "DEFECT SUMMARY:\n"
    total_defects = sum(defect['count'] for defect in defect_results)
    report += f"Total defects found: {total_defects}\n\n"

    report += "DETAILED ANALYSIS:\n"
    for defect in defect_results:
        report += f"- {defect['type']}: {defect['count']} defects ({defect['confidence']} confidence)\n"

    report += "\nRECOMMENDATIONS:\n"
    report += "- Review PCB manufacturing process\n"
    report += "- Check for systematic issues\n"
    report += "- Implement quality control measures\n"

    return report


def generate_sample_image():
    """Generate a sample annotated image for download"""
    # Create a simple sample image
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.text(0.5, 0.5, "CircuitGuard Defect Analysis\n\nDefects Highlighted in Red",
            ha='center', va='center', fontsize=16, transform=ax.transAxes)
    ax.set_facecolor('lightgray')
    ax.axis('off')

    # Save to bytes
    from io import BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='jpg', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return buf


if __name__ == "__main__":
    main()
