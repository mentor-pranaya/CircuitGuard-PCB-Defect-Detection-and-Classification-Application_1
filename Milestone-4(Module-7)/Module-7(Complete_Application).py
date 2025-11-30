import streamlit as st
import torch
import torch.nn.functional as F
import timm
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import os
import glob
from datetime import datetime
import time

# ==============================
# CONFIG
# ==============================
MODEL_PATH = r"D:\CircuitGuard-PCB-Project\Data\best_efficientnet_b4.pth"
TEMPLATE_DIR = r"D:\CircuitGuard-PCB-Project\Data\PCB_DATASET\templates"

OUTPUT_DIR = "module7_outputs"
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)

CLASS_NAMES = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper"
]

TARGET_SIZE = (128, 128)
MIN_MATCH_COUNT = 10        # for homography
MIN_DEFECT_AREA = 25        # remove tiny blobs

# ==============================
# UI CONFIGURATION & CSS
# ==============================
st.set_page_config(
    page_title="CircuitGuard AI | PCB Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_custom_css():
    st.markdown(
        """
        <style>
        /* --- MAIN THEME: Deep Space & Neon --- */
        .stApp {
            background-color: #020617; /* Very dark slate */
            color: #e2e8f0;
        }
        
        /* --- SIDEBAR --- */
        [data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 1px solid #1e293b;
        }
        
        /* Increase Sidebar Heading Sizes */
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            font-size: 1.8rem !important;
            font-weight: 800 !important;
            color: #38bdf8 !important;
            margin-bottom: 0.5rem;
        }
        
        /* Tighter Spacing for Sidebar Text */
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
            font-size: 1.1rem;
            line-height: 1.4;
            margin-bottom: 5px;
        }

        /* --- CARDS / CONTAINERS --- */
        .css-card {
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
        }
        
        /* --- ANIMATED GRADIENT BUTTON --- */
        div.stButton > button {
            background: linear-gradient(45deg, #2563eb, #06b6d4);
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 6px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(6, 182, 212, 0.4);
        }
        
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(6, 182, 212, 0.6);
            background: linear-gradient(45deg, #1d4ed8, #0891b2);
        }

        /* --- HIGHLIGHT BANNER (Dynamic) --- */
        .result-banner {
            background: linear-gradient(90deg, #3730a3 0%, #312e81 100%);
            border-left: 5px solid #818cf8;
            padding: 20px;
            border-radius: 4px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }
        .result-title {
            font-size: 1.5rem;
            font-weight: 800;
            color: #c7d2fe;
            margin: 0;
        }
        .result-conf {
            font-size: 1rem;
            color: #818cf8;
            font-family: monospace;
        }

        /* --- HEADINGS --- */
        h1 {
            font-family: 'Helvetica Neue', sans-serif;
            background: -webkit-linear-gradient(0deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }
        h2, h3 {
            color: #94a3b8;
        }
        
        /* --- METRICS --- */
        [data-testid="stMetricValue"] {
            color: #38bdf8 !important;
        }
        
        /* --- TABLES --- */
        [data-testid="stDataFrame"] {
            border: 1px solid #334155;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ==============================
# CORE LOGIC (UNCHANGED)
# ==============================
@st.cache_resource
def load_model():
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=len(CLASS_NAMES))
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def preprocess_roi(roi_pil: Image.Image) -> torch.Tensor:
    roi_pil = roi_pil.resize(TARGET_SIZE)
    img = np.array(roi_pil).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return tensor

def load_templates(template_dir):
    pattern = os.path.join(template_dir, "*.*")
    paths = [p for p in glob.glob(pattern) if p.lower().endswith((".jpg", ".jpeg", ".png"))]
    templates = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        templates.append((p, img))
    return templates

def choose_best_template(upload_bgr, templates):
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(cv2.cvtColor(upload_bgr, cv2.COLOR_BGR2GRAY), None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    best_score = -1
    best_tpl = None
    best_H = None
    best_aligned = None

    for tpl_path, tpl_bgr in templates:
        gray_tpl = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
        kp2, des2 = orb.detectAndCompute(gray_tpl, None)
        if des2 is None or des1 is None:
            continue
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        score = len(good)
        if score > best_score:
            best_score = score
            best_tpl = (tpl_path, tpl_bgr)
            if score >= MIN_MATCH_COUNT:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H is not None:
                    h, w = gray_tpl.shape
                    aligned = cv2.warpPerspective(upload_bgr, H, (w, h))
                    best_H = H
                    best_aligned = aligned
                else:
                    best_H = None
                    best_aligned = cv2.resize(upload_bgr, (tpl_bgr.shape[1], tpl_bgr.shape[0]))
            else:
                best_H = None
                best_aligned = cv2.resize(upload_bgr, (tpl_bgr.shape[1], tpl_bgr.shape[0]))

    if best_tpl is None:
        return None, None, None, None
    return best_tpl[0], best_tpl[1], best_H, best_aligned

def extract_defect_rois(template_bgr, aligned_bgr):
    tpl_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    tpl_blur = cv2.GaussianBlur(tpl_gray, (5, 5), 0)
    test_blur = cv2.GaussianBlur(test_gray, (5, 5), 0)
    diff = cv2.absdiff(tpl_blur, test_blur)
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    rois = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < MIN_DEFECT_AREA:
            continue
        pad = 3
        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x + w + pad, aligned_bgr.shape[1])
        y1 = min(y + h + pad, aligned_bgr.shape[0])
        roi_bgr = aligned_bgr[y0:y1, x0:x1]
        if roi_bgr.size == 0:
            continue
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        roi_pil = Image.fromarray(roi_rgb)
        rois.append((roi_pil, (x0, y0, x1 - x0, y1 - y0)))
    return rois, thresh

def classify_rois(model, roi_list):
    if not roi_list:
        return []
    tensors = []
    for roi_pil, bbox in roi_list:
        t = preprocess_roi(roi_pil)
        tensors.append(t)
    batch = torch.cat(tensors, dim=0)
    with torch.no_grad():
        outputs = model(batch)
        probs = F.softmax(outputs, dim=1).cpu().numpy()
    results = []
    for (roi_pil, bbox), p in zip(roi_list, probs):
        idx = np.argsort(p)[::-1]
        top1, top2 = idx[0], idx[1] if len(idx) > 1 else idx[0]
        result = {
            "bbox": bbox,
            "top1_class": CLASS_NAMES[top1],
            "top1_conf": float(p[top1]),
            "top2_class": CLASS_NAMES[top2],
            "top2_conf": float(p[top2])
        }
        results.append(result)
    return results

CLASS_COLORS = {
    "Missing_hole": (0, 255, 255),      # yellow
    "Mouse_bite": (255, 0, 255),        # magenta
    "Open_circuit": (0, 165, 255),      # orange
    "Short": (0, 0, 255),               # red
    "Spur": (0, 255, 0),                # green
    "Spurious_copper": (255, 0, 0)      # blue-ish
}

def annotate_board(aligned_bgr, results):
    annotated = aligned_bgr.copy()
    for res in results:
        x, y, w, h = res["bbox"]
        cls = res["top1_class"]
        conf = res["top1_conf"]
        color = CLASS_COLORS.get(cls, (0, 255, 0))
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        label_text = f"{cls}"
        cv2.putText(annotated, label_text, (x, max(y - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_rgb)

# ==============================
# MAIN APPLICATION
# ==============================
def main():
    inject_custom_css()

    # --- SIDEBAR ---
    with st.sidebar:
        # Increased size via CSS, Added Emoji
        st.markdown("## 📟 CircuitGuard")
        st.write("Full Board Analysis")
        st.info("System Ready | EfficientNet-B4")
        
        st.markdown("### ℹ️ How to use")
        st.markdown(
            """
            * **Upload** a PCB Image.
            * **Click** 'Analyze PCB'.
            * **Review** Detected Defects.
            * **Download** Final Report.
            """
        )

    # --- HEADER ---
    st.markdown("<h1>CircuitGuard <span>PCB Analysis</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b;'>A PCB Defect Detection System</p>", unsafe_allow_html=True)

    # --- INPUT SECTION ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🗁 Upload Image")
        uploaded = st.file_uploader("Select PCB Image", type=["jpg", "jpeg", "png"])
        
    if uploaded:
        # Load Image
        bytes_data = uploaded.read()
        np_arr = np.frombuffer(bytes_data, np.uint8)
        upload_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        upload_pil = Image.fromarray(cv2.cvtColor(upload_bgr, cv2.COLOR_BGR2RGB))

        with col1:
            # FIX: use_container_width for compatibility
            st.image(upload_pil, caption="Source", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("### ⚙️ Dashboard")
            run_btn = st.button("Analyze PCB")
            st.markdown("</div>", unsafe_allow_html=True)

            if run_btn:
                # --- PROCESSING PIPELINE ---
                # Replaced status container with spinner to remove blank bars
                with st.spinner("🔍 Processing... (Loading Model, Matching Templates, Scanning...)"):
                    
                    model = load_model()
                    
                    templates = load_templates(TEMPLATE_DIR)
                    if not templates:
                        st.error(f"No templates found in {TEMPLATE_DIR}")
                        return

                    tpl_path, tpl_bgr, H, aligned_bgr = choose_best_template(upload_bgr, templates)
                    if aligned_bgr is None:
                        st.error("Alignment Failed.")
                        return

                    roi_list, mask = extract_defect_rois(tpl_bgr, aligned_bgr)
                    
                    results = classify_rois(model, roi_list)

                    annotated_pil = annotate_board(aligned_bgr, results)

                    # Save
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_name = os.path.splitext(uploaded.name)[0]
                    annotated_path = os.path.join(ANNOTATED_DIR, f"{base_name}_annotated_{timestamp}.png")
                    annotated_pil.save(annotated_path)
                    
                    # CSV Log (Save ALL)
                    log_path = os.path.join(OUTPUT_DIR, "module7_fullboard_log.csv")
                    rows = []
                    for i, res in enumerate(results):
                        x, y, w, h = res["bbox"]
                        rows.append({
                            "upload_filename": uploaded.name,
                            "template_used": os.path.basename(tpl_path),
                            "defect_index": i,
                            "x": x, "y": y, "w": w, "h": h,
                            "top1_class": res["top1_class"],
                            "top1_conf": round(res["top1_conf"], 4),
                            "top2_class": res["top2_class"],
                            "top2_conf": round(res["top2_conf"], 4),
                            "timestamp": timestamp
                        })
                    
                    df_log = pd.DataFrame(rows)
                    if os.path.exists(log_path):
                        df_log.to_csv(log_path, mode="a", header=False, index=False)
                    else:
                        df_log.to_csv(log_path, index=False)

                st.success("✅ Analysis Complete")

                # --- RESULT BANNER (Dynamic) ---
                if results:
                    best_def = max(results, key=lambda x: x['top1_conf'])
                    # Clean banner text
                    st.markdown(f"""
                    <div class='result-banner'>
                        <p class='result-title'>⚠️ Primary Defect: {best_def['top1_class']}</p>
                        <p class='result-conf'>Confidence Score: {best_def['top1_conf']:.2%} | Location: {best_def['bbox']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("✅ No significant defects detected.")

                # --- VISUALS ---
                c_viz1, c_viz2 = st.columns(2)
                with c_viz1:
                    st.markdown("**Aligned Input**")
                    st.image(aligned_bgr, channels="BGR", use_container_width=True)
                with c_viz2:
                    st.markdown("**Annotated Output**")
                    st.image(annotated_pil, use_container_width=True)

                # --- DATA TABLE ---
                if results:
                    with st.expander("Show Detection Data", expanded=False):
                        df_show = pd.DataFrame([
                            {
                                "Type": r["top1_class"],
                                "Confidence": r["top1_conf"],
                                "Secondary": r["top2_class"]
                            } for i, r in enumerate(results)
                        ])
                        df_show = df_show.sort_values(by="Confidence", ascending=False)
                        
                        try:
                            st.dataframe(
                                df_show, 
                                use_container_width=True,
                                column_config={
                                    "Confidence": st.column_config.ProgressColumn(
                                        "Confidence",
                                        format="%.2f",
                                        min_value=0,
                                        max_value=1,
                                    )
                                }
                            )
                        except:
                            st.dataframe(df_show)

                # --- DOWNLOADS ---
                st.markdown("---")
                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    with open(annotated_path, "rb") as f_img:
                        st.download_button(
                            "📥 Download Image",
                            data=f_img.read(),
                            file_name=os.path.basename(annotated_path),
                            mime="image/png"
                        )
                with d_col2:
                    with open(log_path, "rb") as f_log:
                        st.download_button(
                            "📥 Download CSV Log",
                            data=f_log.read(),
                            file_name="module7_fullboard_log.csv",
                            mime="text/csv"
                        )

if __name__ == "__main__":
    main()