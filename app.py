import streamlit as st
import torch
from torchvision import transforms
from timm import create_model
from PIL import Image

st.set_page_config(
    page_title="CircuitGuard PCB Defect Detection",
    layout="centered",
)

st.markdown("""
<style>

html, body {
    font-family: 'Segoe UI', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0c0f17, #041424);
    padding: 0;
    margin: 0;
}

.header-box {
    text-align: center;
    padding: 25px 10px;
    margin-bottom: 20px;
    background: linear-gradient(90deg, #32cd71, #205f35);
    border-radius: 12px;
    color: white;
    animation: glow 3s infinite alternate;
}

@keyframes glow {
    0% { box-shadow: 0 0 8px #32cd71; }
    100% { box-shadow: 0 0 20px #32cd71; }
}

.upload-card {
    background: rgba(255, 255, 255, 0.06);
    padding: 20px;
    border-radius: 18px;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    transition: 0.3s ease;
}
.upload-card:hover {
    transform: translateY(-3px);
}

.result-card {
    background: rgba(76, 175, 80, 0.10);
    padding: 20px;
    border-radius: 12px;
    border: 1px solid rgba(76, 175, 80, 0.35);
    margin-top: 20px;
}

.conf-card {
    background: rgba(100, 149, 237, 0.10);
    padding: 15px;
    border-radius: 12px;
    border: 1px solid rgba(100, 149, 237, 0.35);
    margin-top: 10px;
}

.pred-label {
    font-size: 28px;
    color: #32cd71;
    font-weight: 700;
    text-shadow: 0 0 10px rgba(50, 205, 113, 0.5);
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='header-box'>
    <h1 style='margin-bottom:-5px;'>⚙️ CircuitGuard</h1>
    <p>AI-Powered PCB Defect Detection System</p>
</div>
""", unsafe_allow_html=True)

DEVICE = "cpu"
MODEL_PATH = "Data/best_efficientnet_b4.pth"

CLASS_NAMES = [
    'Missing_hole',
    'Mouse_bite',
    'Open_circuit',
    'Short',
    'Spur',
    'Spurious_copper'
]

@st.cache_resource
def load_model():
    model = create_model('efficientnet_b4', pretrained=False, num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

st.markdown("<div class='upload-card'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(" Upload Image", type=["png", "jpg", "jpeg"])
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file:

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    analyze_btn = st.button("Analyze Defect", use_container_width=True)

    if analyze_btn:

        with st.spinner("Processing.."):

            input_tensor = transform(img).unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0]

            conf, idx = torch.max(probs, 0)
            predicted_label = CLASS_NAMES[idx.item()]

            st.markdown(f"""
            <div class='result-card'>
                <h2 class='pred-label'>Predicted Defect: {predicted_label}</h2>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class='conf-card'>
                <h4 style='color:#6FB1FF;'>Confidence: {conf.item()*100:.2f}%</h4>
            </div>
            """, unsafe_allow_html=True)

            st.subheader("📊 Class Probabilities")
            for i, cls in enumerate(CLASS_NAMES):
                st.write(f"**{cls}** — {probs[i].item()*100:.2f}%")

else:
    st.info("Upload an image to begin analysis.")