import torch
import timm
from torchvision import transforms
from PIL import Image
import cv2

# paths
test_img_path = "test/test.jpg"
golden_pcb_path = "test/golden.jpg"
final_img_path = "test/img_final.jpg"

# class labels (same order used during training)
classes = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# preprocessing for the classifier
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# load model
model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=len(classes))
model.load_state_dict(torch.load("efficientnet_b4_pcb.pth", map_storage=device))
model.to(device)
model.eval()

def classify_roi(roi):
    # convert grayscale to rgb if needed
    if len(roi.shape) == 2:
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)

    roi = Image.fromarray(roi)
    t = test_transform(roi).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(t)

    return classes[out.argmax().item()]


# load main images
img = cv2.imread(test_img_path)
img_BW = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

golden = cv2.imread(golden_pcb_path)
golden_BW = cv2.cvtColor(golden, cv2.COLOR_BGR2GRAY)

# basic diff-based defect detection
sub = cv2.absdiff(golden_BW, img_BW)
_, thresh = cv2.threshold(sub, 20, 255, cv2.THRESH_BINARY)

# get contours from diff image
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

pad = 5
min_area = 50
min_size = 10

# draw on a fresh copy
img_final = cv2.imread(test_img_path)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < min_area:
        continue

    x, y, w, h = cv2.boundingRect(cnt)
    if w < min_size or h < min_size:
        continue

    # padded roi bounds
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, img_BW.shape[1])
    y2 = min(y + h + pad, img_BW.shape[0])

    roi = img_BW[y1:y2, x1:x2]

    pred = classify_roi(roi)
    print("Predicted:", pred)

    # defect box
    cv2.rectangle(img_final, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # label background + text
    label = pred
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.7
    th = 2

    tx = x1
    ty = max(y1 - 10, 0)

    (tw, tht), base = cv2.getTextSize(label, font, fs, th)
    
    # Draw background rectangle (filled)
    cv2.rectangle(img_final,
                  (tx, ty - tht - base),
                  (tx + tw, ty + base),
                  (0, 165, 255),
                  -1)

    # Draw text
    cv2.putText(img_final, label, (tx, ty),
                font, fs, (0, 0, 0), th)

# save output
cv2.imwrite(final_img_path, img_final)
