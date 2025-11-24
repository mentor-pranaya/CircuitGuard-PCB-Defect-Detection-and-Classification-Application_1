"""
Module 4: Evaluation and Prediction Testing
CircuitGuard - PCB Defect Detection
Configured for EfficientNet-B0 trained model
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import os
import json
import glob
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
class Config:
    ROOT_DATA_DIR = '.'
    ROIS_DIR = os.path.join(ROOT_DATA_DIR, 'CLASSIFICATION_ROIS_128x128')
    MODEL_PATH = os.path.join(ROOT_DATA_DIR, 'best_efficientnet_b0.pth')
    OUTPUT_DIR = 'MODULE4_RESULTS'
    
    MODEL_NAME = 'efficientnet_b0'
    IMG_SIZE = 128
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    TEST_SPLIT = 0.2
    RANDOM_SEED = 42
    
    CLASS_NAMES = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 
                   'Short', 'Spur', 'Spurious_copper']
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# ==================== Dataset Class ====================
class PCBTestDataset(Dataset):
    def __init__(self, file_list, labels, label_map, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.label_map = label_map
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.label_map[self.labels[idx]]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path

# ==================== Load Data ====================
def load_data_paths():
    all_files = glob.glob(os.path.join(Config.ROIS_DIR, '*', '*.png'))
    all_files.extend(glob.glob(os.path.join(Config.ROIS_DIR, '*', '*.jpg')))
    
    all_labels = [os.path.basename(os.path.dirname(f)) for f in all_files]
    
    print(f"\nLoaded {len(all_files)} images from {Config.ROIS_DIR}")
    
    from collections import Counter
    label_counts = Counter(all_labels)
    print("\nImages per class:")
    for label in sorted(label_counts.keys()):
        print(f"{label:20s}: {label_counts[label]:4d}")
    
    return all_files, all_labels

# ==================== Model Loading ====================
def load_model(model_path, num_classes, device):
    print(f"\nLoading model from {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = timm.create_model(Config.MODEL_NAME, pretrained=False, num_classes=num_classes)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully.")
    return model

# ==================== Inference Pipeline ====================
def run_inference(model, dataloader, device):
    all_preds = []
    all_labels = []
    all_paths = []
    
    print("\nRunning inference...")
    
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
    
    return np.array(all_preds), np.array(all_labels), all_paths

# ==================== Evaluation Metrics ====================
def calculate_metrics(y_true, y_pred, class_names):
    accuracy = accuracy_score(y_true, y_pred)
    
    print("\nAccuracy:", accuracy * 100)
    
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names, 
        digits=4
    )
    print("\nClassification Report:\n")
    print(report)
    
    report_dict = classification_report(
        y_true, y_pred, 
        target_names=class_names, 
        output_dict=True
    )
    
    report_dict['overall_accuracy'] = float(accuracy)
    
    metrics_path = os.path.join(Config.OUTPUT_DIR, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(report_dict, f, indent=4)
    
    print(f"Metrics saved to: {metrics_path}")
    return accuracy, report_dict

# ==================== Confusion Matrix ====================
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(14, 10))
    
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    sns.heatmap(
        cm, annot=annot, fmt='', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Confusion Matrix - PCB Defect Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = os.path.join(Config.OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    
    print(f"Confusion matrix saved to: {cm_path}")
    plt.show()
    
    return cm

# ==================== Per-Class Accuracy ====================
def plot_per_class_accuracy(y_true, y_pred, class_names):
    accuracies = []
    for i in range(len(class_names)):
        mask = y_true == i
        if mask.sum() > 0:
            accuracies.append((y_pred[mask] == y_true[mask]).mean() * 100)
        else:
            accuracies.append(0)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, accuracies)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%', 
                 ha='center', va='bottom')
    
    plt.xlabel('Defect Type')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Classification Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    acc_path = os.path.join(Config.OUTPUT_DIR, 'per_class_accuracy.png')
    plt.savefig(acc_path, dpi=300, bbox_inches='tight')
    
    print(f"Per-class accuracy plot saved to: {acc_path}")
    plt.show()

# ==================== Prediction Visualization ====================
def visualize_predictions(paths, y_true, y_pred, class_names, num_samples=16):
    indices = np.random.choice(len(paths), min(num_samples, len(paths)), replace=False)
    
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        img_path = paths[idx]
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        
        try:
            img = Image.open(img_path).convert('RGB')
            axes[i].imshow(img)
            color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
        except:
            axes[i].text(0.5, 0.5, 'Image not found', ha='center', va='center')
        
        axes[i].axis('off')
    
    plt.tight_layout()
    
    viz_path = os.path.join(Config.OUTPUT_DIR, 'prediction_samples.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    
    print(f"Prediction visualization saved to: {viz_path}")
    plt.show()

# ==================== Error Analysis ====================
def analyze_errors(paths, y_true, y_pred, class_names):
    errors = y_true != y_pred
    error_indices = np.where(errors)[0]
    
    print(f"\nTotal Misclassifications: {len(error_indices)}")
    
    error_analysis = {}
    for idx in error_indices:
        key = f"{class_names[y_true[idx]]} -> {class_names[y_pred[idx]]}"
        error_analysis[key] = error_analysis.get(key, 0) + 1
    
    sorted_errors = sorted(error_analysis.items(), key=lambda x: x[1], reverse=True)
    
    print("\nMost Common Misclassification Patterns:")
    for pattern, count in sorted_errors[:10]:
        print(f"{pattern}: {count}")

# ==================== Main Pipeline ====================
def main():
    print("Module 4: Evaluation and Prediction Testing")
    print(f"Device: {Config.DEVICE}")
    
    np.random.seed(Config.RANDOM_SEED)
    torch.manual_seed(Config.RANDOM_SEED)
    
    test_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    all_files, all_labels = load_data_paths()
    if len(all_files) == 0:
        print("No images found.")
        return
    
    label_map = {label: i for i, label in enumerate(sorted(set(all_labels)))}
    class_names = list(label_map.keys())
    num_classes = len(class_names)
    
    _, test_files, _, test_labels = train_test_split(
        all_files, all_labels, 
        test_size=Config.TEST_SPLIT, 
        stratify=all_labels,
        random_state=Config.RANDOM_SEED
    )
    
    print(f"Test set size: {len(test_files)}")
    
    test_dataset = PCBTestDataset(test_files, test_labels, label_map, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                             num_workers=Config.NUM_WORKERS)
    
    model = load_model(Config.MODEL_PATH, num_classes, Config.DEVICE)
    
    y_pred, y_true, paths = run_inference(model, test_loader, Config.DEVICE)
    
    accuracy, metrics = calculate_metrics(y_true, y_pred, class_names)
    
    cm = plot_confusion_matrix(y_true, y_pred, class_names)
    
    plot_per_class_accuracy(y_true, y_pred, class_names)
    
    visualize_predictions(paths, y_true, y_pred, class_names)
    
    analyze_errors(paths, y_true, y_pred, class_names)
    
    print("\nEvaluation complete.")
    print("Results saved to:", Config.OUTPUT_DIR)

if __name__ == "__main__":
    main()

