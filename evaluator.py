import os
import cv2
import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from models.classifier import load_model_checkpoint
from utils.file_utils import ensure_dir

def evaluate_on_folder(model_path: str, test_folder: str, model_name: str = "efficientnet_b0", device: str = "cpu"):
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    test_ds = datasets.ImageFolder(test_folder, transform=transform)
    loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)
    class_names = test_ds.classes
    num_classes = len(class_names)

    model = load_model_checkpoint(model_path, model_name=model_name, num_classes=num_classes, device=device)

    all_preds = []
    all_labels = []
    out_vis_dir = os.path.join("outputs", "eval")
    ensure_dir(out_vis_dir)

    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

            # save simple annotated images
            imgs_cpu = imgs.cpu().numpy()
            for i in range(imgs_cpu.shape[0]):
                img_vis = imgs_cpu[i].transpose(1,2,0)
                img_vis = (img_vis * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]))
                img_vis = (img_vis * 255).clip(0,255).astype(np.uint8)
                gt = class_names[labels[i].item()]
                pred = class_names[preds[i]]
                color = (0,255,0) if gt==pred else (0,0,255)
                img_vis = cv2.resize(img_vis, (400,400))
                cv2.putText(img_vis, f"GT: {gt}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(img_vis, f"Pred: {pred}", (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                fname = os.path.join(out_vis_dir, f"res_{idx}_{i}.jpg")
                cv2.imwrite(fname, cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))

    report = classification_report(all_labels, all_preds, target_names=class_names)
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8,6))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45)
    return report, fig, out_vis_dir
