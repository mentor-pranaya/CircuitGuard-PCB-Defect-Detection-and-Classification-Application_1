#In this Script we are training the model and generating the confusion Matrix on testing 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
import os

# Paths
DATA_DIR = r"C:/Users/Badal/Desktop/SpringBoard/Model_Data/ROI"
MODEL_SAVE_PATH = "efficientnet_b4_pcb.pth"

# Image size
IMG_SIZE = 128
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4

# ---------------------------------------
# TRANSFORMS (Augmentations + Normalize)
# ---------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# ---------------------------------------
# DATASET & DATALOADERS
# ---------------------------------------

train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_data   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=test_transform)
test_data  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE)

num_classes = len(train_data.classes)
print("Classes:", train_data.classes)

# ---------------------------------------
# MODEL — EfficientNet-B4
# ---------------------------------------

model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---------------------------------------
# TRAINING LOOP
# ---------------------------------------

train_acc_list = []
val_acc_list = []
train_loss_list = []
val_loss_list = []

for epoch in range(NUM_EPOCHS):
    model.train()
    total, correct, total_loss = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = correct / total
    train_loss = total_loss / len(train_loader)

    # ----- VALIDATION -----
    model.eval()
    total, correct, val_loss_total = 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_total += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_acc = correct / total
    val_loss = val_loss_total / len(val_loader)

    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

# Save model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("Model saved:", MODEL_SAVE_PATH)

# ---------------------------------------
# PLOT ACCURACY & LOSS
# ---------------------------------------

plt.plot(train_acc_list, label="Train Acc")
plt.plot(val_acc_list, label="Val Acc")
plt.legend()
plt.title("Accuracy Curve")
plt.show()

plt.plot(train_loss_list, label="Train Loss")
plt.plot(val_loss_list, label="Val Loss")
plt.legend()
plt.title("Loss Curve")
plt.show()

# ---------------------------------------
# CONFUSION MATRIX (TEST SET)
# ---------------------------------------

model.eval()
true_labels = []
pred_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        
        true_labels.extend(labels.numpy())
        pred_labels.extend(predicted.cpu().numpy())

cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=train_data.classes,
            yticklabels=train_data.classes)
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:")
print(classification_report(true_labels, pred_labels,
                           target_names=train_data.classes))
