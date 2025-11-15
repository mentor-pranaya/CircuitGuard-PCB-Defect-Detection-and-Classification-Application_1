import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import timm
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DATA_DIR = "../Data"
ROIS_DIR = os.path.join(ROOT_DATA_DIR, "CLASSIFICATION_ROIS_128x128")
MODEL_SAVE_PATH = os.path.join(ROOT_DATA_DIR, "best_efficientnet_b4.pth")

TARGET_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4


# =============================================
# DATASET
# =============================================
class PCBDefectDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

        # Create mapping like ImageFolder
        self.label_map = {label: i for i, label in enumerate(sorted(set(labels)))}
        self.num_classes = len(self.label_map)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.label_map[self.labels[idx]]

        if self.transform:
            image = self.transform(image)

        return image, label



def load_data_paths():
    all_files = glob.glob(os.path.join(ROIS_DIR, "*", "*.png"))
    all_labels = [os.path.basename(os.path.dirname(f)) for f in all_files]
    return all_files, all_labels


def get_transforms(is_train):
    if is_train:
        return transforms.Compose([
            transforms.Resize(TARGET_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(TARGET_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs):
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(f"🚀 Training on {DEVICE} for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_corrects, total_train = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_corrects += (outputs.argmax(1) == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / total_train
        train_acc = running_corrects / total_train

        # ========= Validation ==========
        model.eval()
        val_loss, val_correct, total_val = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                total_val += labels.size(0)

        val_loss /= total_val
        val_acc = val_correct / total_val
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc * 100)

        scheduler.step(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"💾 Model saved with Val Acc: {best_acc*100:.2f}%")

    return history, best_acc



def evaluate_model(model, loader):
    model.eval()
    preds, labs = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds.extend(outputs.argmax(1).cpu().numpy())
            labs.extend(labels.cpu().numpy())

    return labs, preds

def plot_history(history):
    plt.figure(figsize=(10, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(history["val_acc"], label="Val Accuracy")
    plt.legend()
    plt.title("Validation Accuracy (%)")
    plt.show()


def plot_confusion_matrix(true_classes, pred_classes, class_names):
    cm = confusion_matrix(true_classes, pred_classes)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


# =============================================
# MAIN
# =============================================
files, labels = load_data_paths()

if not files:
    print(f"❌ No ROI images found in: {ROIS_DIR}")
    exit()

# Train/Test split
train_f, val_f, train_l, val_l = train_test_split(
    files, labels, test_size=0.2, stratify=labels, random_state=42
)

train_ds = PCBDefectDataset(train_f, train_l, transform=get_transforms(True))
val_ds = PCBDefectDataset(val_f, val_l, transform=get_transforms(False))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

num_classes = train_ds.num_classes
class_names = sorted(set(labels))

print(f"📂 Training samples: {len(train_ds)} | Classes: {num_classes}")


model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=num_classes)


for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if "blocks.6" in name or "blocks.7" in name or "classifier" in name:
        param.requires_grad = True

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

# =============================================
# TRAIN
# =============================================
history, best_acc = train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, NUM_EPOCHS)

print(f"\n🏆 Best Validation Accuracy: {best_acc*100:.2f}%")

plot_history(history)

model.load_state_dict(torch.load(MODEL_SAVE_PATH))

true_l, pred_l = evaluate_model(model, val_loader)

plot_confusion_matrix(true_l, pred_l, class_names)
