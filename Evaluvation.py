import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import timm

# =============== SETTINGS ===============
test_dir = "../Data/CLASSIFICATION_ROIS_128x128"
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", device)

# SAME TRANSFORMS AS TRAINING (NO AUGMENTATION)
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load dataset
test_data = datasets.ImageFolder(test_dir, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

class_names = test_data.classes
num_classes = len(class_names)
print("Classes:", class_names)

# =============== LOAD MODEL CORRECTLY ===============
model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load("../Data/best_efficientnet_b4.pth", map_location=device))
model.to(device)
model.eval()

# =============== PREDICTION ===============
all_preds = []
all_labels = []

os.makedirs("Output_Test_Results", exist_ok=True)

with torch.no_grad():
    for idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # ---- Save visualization images ----
        for i in range(images.size(0)):
            img_vis = images[i].cpu().numpy().transpose(1, 2, 0)
            img_vis = (img_vis * 0.229 + 0.485)  # unnormalize
            img_vis = (img_vis * 255).clip(0, 255).astype(np.uint8)
            img_vis = cv2.resize(img_vis, (400, 400))

            gt = class_names[labels[i].item()]
            pred = class_names[preds[i].item()]

            color = (0, 255, 0) if gt == pred else (0, 0, 255)
            cv2.putText(img_vis, f"GT: {gt}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(img_vis, f"Pred: {pred}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imwrite(f"Output_Test_Results/result_{idx}_{i}.jpg", img_vis)

print("Annotated prediction images saved in Output_Test_Results/")

# =============== METRICS ===============
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=class_names))

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
plt.figure(figsize=(8, 6))
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix - PCB Defect Classifier")
plt.show()

acc = np.mean(np.array(all_labels) == np.array(all_preds))
print(f"\nFinal Accuracy: {acc * 100:.2f}%")
