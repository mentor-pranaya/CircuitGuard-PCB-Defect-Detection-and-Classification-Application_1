import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
import os
import glob
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# Configuration
DATA_DIR = "DEFECT_CROPS"
MODEL_SAVE_PATH = "best_efficientnet_b4.pth"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001

# Custom dataset loader for PCB defects
class PCBDefectDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(root_dir, "*.jpg"))
        self.transform = transform
        self.labels = []
        self.clean_paths = []

        # Expected defect categories
        self.classes = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']

        print(f"Scanning {root_dir}...")
        print(f"Found {len(self.image_paths)} total files.")

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in '{root_dir}'. Ensure it contains .jpg files.")

        # Match filenames to labels
        for path in self.image_paths:
            filename = os.path.basename(path).lower()
            found_label = None

            for defect_name in self.classes:
                if defect_name in filename:
                    found_label = defect_name
                    break

            if found_label:
                self.labels.append(found_label)
                self.clean_paths.append(path)

        if len(self.clean_paths) == 0:
            raise ValueError("Images found, but none matched expected defect class names. Check filenames.")

        # Encode labels
        self.encoder = LabelEncoder()
        self.encoded_labels = self.encoder.fit_transform(self.labels)
        print(f"Loaded {len(self.clean_paths)} valid images.")
        print(f"Classes found: {self.encoder.classes_}")

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        img_path = self.clean_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            print(f"Warning: Could not open {img_path}. Returning dummy tensor.")
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), 0, img_path

        label = self.encoded_labels[idx]
        if self.transform:
            image = self.transform(image)

        return image, label, img_path


def train_efficientnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    try:
        dataset = PCBDefectDataset(DATA_DIR, transform=transform)
    except ValueError as e:
        print(e)
        return

    # Train-validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    if len(train_data) == 0:
        print("Error: Training dataset is empty.")
        return

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # Build model
    print("Building EfficientNet-B4 model...")
    model = models.efficientnet_b4(weights='DEFAULT')

    # Update classifier
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    print("Starting training...")

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total if total > 0 else 0
        avg_loss = running_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save model if accuracy improves
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"Training complete. Best model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_efficientnet()

