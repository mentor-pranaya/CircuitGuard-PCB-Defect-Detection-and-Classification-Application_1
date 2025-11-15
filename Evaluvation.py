import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

test_dir = "dataset/test"
batch_size = 32
num_classes = 6
class_names = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_data = datasets.ImageFolder(test_dir, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


model = torch.load("best_pcb_model.pth", map_location=device)
model.to(device)
model.eval()

all_preds = []
all_labels = []

# Create folder for output predictions
os.makedirs("Output_Test_Results", exist_ok=True)

with torch.no_grad():
    for idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # ---------- Save annotated images ----------
        for i in range(images.shape[0]):
            img_np = images[i].cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)
            img_np = cv2.resize(img_np, (400, 400))

            gt = class_names[labels[i].item()]
            pred = class_names[preds[i].item()]

            # Add predicted + true label text
            txt_color = (0, 255, 0) if gt == pred else (0, 0, 255)
            cv2.putText(img_np, f"GT: {gt}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(img_np, f"Pred: {pred}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_color, 2)

            cv2.imwrite(f"Output_Test_Results/output_{idx}_{i}.jpg", img_np)

print("Annotated prediction images saved in Output_Test_Results/")


print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=class_names))

# CONFUSION MATRIX
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
plt.figure(figsize=(8, 6))
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix - PCB Defect Classifier")
plt.show()

# ACCURACY
accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
print(f"\nFinal Accuracy: {accuracy * 100:.2f}%")
