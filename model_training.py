import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import timm
import os
import cv2
from tqdm import tqdm


class PCBDefectDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # Get class names
        self.class_names = [d for d in os.listdir(data_dir)
                            if os.path.isdir(os.path.join(data_dir, d))]
        self.class_names.sort()
        self.class_to_idx = {cls_name: i for i,
                             cls_name in enumerate(self.class_names)}

        # Load images
        for class_name in self.class_names:
            class_dir = os.path.join(data_dir, class_name, "cropped_defects")
            if not os.path.exists(class_dir):
                continue

            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

        print(
            f"📊 Loaded {len(self.images)} images across {len(self.class_names)} classes")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label


def main():
    print("🚀 Starting Module 3: EfficientNet Model Training")

    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Data transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    data_dir = "results/module2_output"
    dataset = PCBDefectDataset(data_dir, transform=transform)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create model
    model = timm.create_model(
        'efficientnet_b0', pretrained=True, num_classes=len(dataset.class_names))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("🏋️ Starting training...")
    for epoch in range(5):  # Short training for testing
        model.train()
        train_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {train_loss/len(train_loader):.4f}')

    # Save model
    torch.save(model.state_dict(), 'results/pcb_defect_model.pth')
    print("💾 Model saved as 'results/pcb_defect_model.pth'")


if __name__ == "__main__":
    main()
