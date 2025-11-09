#This Script takes the already created model and classifies some unseen images 
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -----------------------------
# PATHS
# -----------------------------
DATA_DIR = r"C:/Users/Badal/Desktop/SpringBoard/Model_Data/ROI/test"
MODEL_PATH = r"C:/Users/Badal/Desktop/SpringBoard/efficientnet_b4_pcb.pth"
OUTPUT_CSV = "pcb_predictions.csv"

IMG_SIZE = 128
BATCH_SIZE = 32

# -----------------------------
# TRANSFORMS
# -----------------------------
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# -----------------------------
# LOAD DATASET
# -----------------------------
test_dataset = datasets.ImageFolder(DATA_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

classes = test_dataset.classes
print("Classes:", classes)

# -----------------------------
# LOAD MODEL
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("Model loaded")

# -----------------------------
# INFERENCE
# -----------------------------
results = []
sample_paths = [path for path, label in test_dataset.samples]

index = 0
true_labels = []
pred_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)

        _, predicted = outputs.max(1)
        predicted = predicted.cpu().numpy()
        labels = labels.numpy()

        batch_size = len(predicted)

        for i in range(batch_size):
            results.append({
                "file": os.path.basename(sample_paths[index]),
                "true_class": classes[labels[i]],
                "predicted_class": classes[predicted[i]]
            })

            true_labels.append(labels[i])
            pred_labels.append(predicted[i])

            index += 1

# -----------------------------
# SAVE CSV
# -----------------------------
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nPredictions saved to: {OUTPUT_CSV}\n")

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(true_labels, pred_labels)

plt.figure(figsize=(10, 7))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------
print("\nClassification Report:\n")
print(classification_report(true_labels, pred_labels, target_names=classes))
'''
Classification Report:

                 precision    recall  f1-score   support

   missing_hole       1.00      1.00      1.00        76
     mouse_bite       0.99      0.99      0.99        75
   open_circuit       0.96      0.96      0.96        75
          short       1.00      0.97      0.99        76
           spur       0.96      0.99      0.97        75
spurious_copper       0.97      0.97      0.97        76

       accuracy                           0.98       453
      macro avg       0.98      0.98      0.98       453
   weighted avg       0.98      0.98      0.98       453
'''
