import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split

from utils.file_utils import ensure_dir
from utils.data_loader import TARGET_SIZE

def train_loop(model, rois_dir: str, model_save_path: str, epochs: int = 10, batch_size: int = 16, lr: float = 1e-4, device: str = "cpu"):
    """
    Simple training loop using ImageFolder (rois_dir/train & rois_dir/test or just rois_dir)
    Splits data 80/20 if not explicit train/val present.
    Saves model state_dict to model_save_path
    """
    # prepare dataset
    if os.path.exists(os.path.join(rois_dir, "train")) and os.path.exists(os.path.join(rois_dir, "test")):
        train_dir = os.path.join(rois_dir, "train")
        val_dir = os.path.join(rois_dir, "test")
    else:
        # use same folder and split
        train_dir = rois_dir
        val_dir = rois_dir

    transform_train = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(0.1,0.1,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # if separate train/test exist use them; else split
    if train_dir == val_dir:
        all_ds = datasets.ImageFolder(train_dir, transform=transform_train)
        if len(all_ds) < 2:
            raise RuntimeError("Not enough images to split for training.")
        # build indices split
        targets = [s[1] for s in all_ds.samples]
        train_idx, val_idx = train_test_split(range(len(targets)), test_size=0.2, stratify=targets, random_state=42)
        from torch.utils.data import Subset
        train_ds = Subset(all_ds, train_idx)
        val_ds = Subset(datasets.ImageFolder(val_dir, transform=transform_val), val_idx)
    else:
        train_ds = datasets.ImageFolder(train_dir, transform=transform_train)
        val_ds = datasets.ImageFolder(val_dir, transform=transform_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    num_classes = len(train_loader.dataset.dataset.classes) if hasattr(train_loader.dataset, "dataset") else len(train_loader.dataset.classes)
    model.to(device)

    # freeze base then fine-tune head
    for p in model.parameters():
        p.requires_grad = False
    for n,p in model.named_parameters():
        if "blocks.6" in n or "blocks.7" in n or "classifier" in n:
            p.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    ensure_dir(os.path.dirname(model_save_path) or ".")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_train = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / total_train
        train_acc = running_correct / total_train

        # validation
        model.eval()
        val_loss = 0.0
        val_corr = 0
        total_val = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_corr += (outputs.argmax(1) == labels).sum().item()
                total_val += labels.size(0)
        val_loss = val_loss / total_val
        val_acc = val_corr / total_val

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc * 100)

        scheduler.step(val_acc)

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)

    return history, model_save_path
