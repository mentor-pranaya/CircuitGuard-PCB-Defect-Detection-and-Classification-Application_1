import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="CircuitGuard - PCB Defect Detection", layout="wide")

# Title and Description
st.title("🔍 CircuitGuard: PCB Defect Detection System")
st.markdown("""
Upload a **defect-free template image** and a **test PCB image** to detect and highlight defects.
""")

# Create two columns for upload
col1, col2 = st.columns(2)

with col1:
    st.subheader("1️⃣ Upload Defect-Free Template")
    template_file = st.file_uploader("Choose template image", type=['jpg', 'jpeg', 'png'], key="template")
    if template_file:
        template_image = Image.open(template_file)
        st.image(template_image, caption="Template Image", use_container_width=True)

with col2:
    st.subheader("2️⃣ Upload Test PCB Image")
    test_file = st.file_uploader("Choose test image", type=['jpg', 'jpeg', 'png'], key="test")
    if test_file:
        test_image = Image.open(test_file)
        st.image(test_image, caption="Test Image", use_container_width=True)

# Processing Button
if st.button("🚀 Detect Defects", type="primary"):
    if template_file and test_file:
        with st.spinner("Processing images..."):
            # Convert PIL images to OpenCV format
            template_cv = cv2.cvtColor(np.array(template_image), cv2.COLOR_RGB2BGR)
            test_cv = cv2.cvtColor(np.array(test_image), cv2.COLOR_RGB2BGR)

            # Resize test image to match template
            if template_cv.shape != test_cv.shape:
                test_cv = cv2.resize(test_cv, (template_cv.shape[1], template_cv.shape[0]))

            # Compute difference
            difference = cv2.absdiff(template_cv, test_cv)
            gray_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_diff, 10, 255, cv2.THRESH_BINARY)
            
            # Highlight defects in red
            highlighted = test_cv.copy()
            highlighted[mask != 0] = [0, 0, 255]
            
            # Convert back to RGB for display
            difference_rgb = cv2.cvtColor(difference, cv2.COLOR_BGR2RGB)
            highlighted_rgb = cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB)

            st.success("✅ Processing Complete!")

            result_col1, result_col2 = st.columns(2)
            with result_col1:
                st.subheader("📊 Difference Map")
                st.image(difference_rgb, caption="Detected Differences", use_container_width=True)
                diff_pil = Image.fromarray(difference_rgb)
                buf1 = io.BytesIO()
                diff_pil.save(buf1, format='PNG')
                st.download_button(
                    label="⬇️ Download Difference Map",
                    data=buf1.getvalue(),
                    file_name="difference_map.png",
                    mime="image/png"
                )
            with result_col2:
                st.subheader("🎯 Highlighted Defects")
                st.image(highlighted_rgb, caption="Defects Highlighted in Red", use_container_width=True)
                high_pil = Image.fromarray(highlighted_rgb)
                buf2 = io.BytesIO()
                high_pil.save(buf2, format='PNG')
                st.download_button(
                    label="⬇️ Download Highlighted Image",
                    data=buf2.getvalue(),
                    file_name="highlighted_defects.png",
                    mime="image/png"
                )

            # Count defects and extract ROIs
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
            defect_count = len(valid_contours)
            st.metric("🔴 Defects Detected", defect_count)

            if defect_count > 0:
                st.markdown("---")
                st.subheader("📦 Extracted Defect ROIs")
                roi_cols = st.columns(min(4, defect_count))
                for idx, cnt in enumerate(valid_contours[:8]):  # Show max 8 ROIs
                    x, y, w, h = cv2.boundingRect(cnt)
                    roi = highlighted_rgb[y:y+h, x:x+w]
                    col_idx = idx % 4
                    with roi_cols[col_idx]:
                        st.image(roi, caption=f"ROI {idx+1}", use_container_width=True)
                if defect_count > 8:
                    st.info(f"ℹ️ Showing 8 of {defect_count} detected ROIs")
    else:
        st.warning("⚠️ Please upload both template and test images!")

# Footer
st.markdown("---")
st.markdown("**CircuitGuard** - Automated PCB Defect Detection System | Milestone 3: Web Interface")

