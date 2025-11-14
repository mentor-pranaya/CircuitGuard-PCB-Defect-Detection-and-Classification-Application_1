import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import timm
import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import your existing dataset class
from model_training import PCBDefectDataset


def load_trained_model(model_path, num_classes):
    """Load your already trained model"""
    print("🔄 Loading trained model...")

    # Create the same model architecture used during training
    model = timm.create_model(
        'efficientnet_b0', pretrained=False, num_classes=num_classes)

    # Load the trained weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print(f"✅ Model loaded successfully from: {model_path}")
        return model
    else:
        print(f"❌ Model file not found: {model_path}")
        return None


def evaluate_model(model, data_loader, device, class_names):
    """Evaluate the model and calculate accuracy"""
    print("📊 Evaluating model performance...")

    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return all_predictions, all_labels, all_probabilities


def calculate_accuracy(predictions, labels):
    """Calculate overall accuracy"""
    correct = np.sum(np.array(predictions) == np.array(labels))
    total = len(labels)
    accuracy = 100 * correct / total
    return accuracy, correct, total


def plot_confusion_matrix(labels, predictions, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - PCB Defect Classification')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('results/model_evaluation_cm.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_class_accuracy(labels, predictions, class_names):
    """Plot accuracy per class"""
    class_accuracy = []

    for i, class_name in enumerate(class_names):
        class_mask = np.array(labels) == i
        if np.sum(class_mask) > 0:
            class_correct = np.sum(np.array(predictions)[class_mask] == i)
            class_total = np.sum(class_mask)
            class_acc = 100 * class_correct / class_total
            class_accuracy.append(class_acc)
        else:
            class_accuracy.append(0)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, class_accuracy,
                   color='skyblue', edgecolor='black')
    plt.title('Accuracy per Defect Type')
    plt.xlabel('Defect Type')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, acc in zip(bars, class_accuracy):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('results/class_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_detailed_report(labels, predictions, class_names):
    """Print comprehensive evaluation report"""
    accuracy, correct, total = calculate_accuracy(predictions, labels)

    print("\n" + "="*70)
    print("🎯 MODEL EVALUATION REPORT")
    print("="*70)

    print(f"\n📈 OVERALL PERFORMANCE:")
    print(f"   • Total Test Samples: {total}")
    print(f"   • Correct Predictions: {correct}")
    print(f"   • Overall Accuracy: {accuracy:.2f}%")

    print(f"\n📊 DETAILED CLASSIFICATION REPORT:")
    report = classification_report(
        labels, predictions, target_names=class_names, digits=4)
    print(report)

    # Per-class accuracy
    print(f"\n🎯 ACCURACY PER DEFECT TYPE:")
    for i, class_name in enumerate(class_names):
        class_mask = np.array(labels) == i
        if np.sum(class_mask) > 0:
            class_correct = np.sum(np.array(predictions)[class_mask] == i)
            class_total = np.sum(class_mask)
            class_acc = 100 * class_correct / class_total
            print(
                f"   • {class_name}: {class_acc:.2f}% ({class_correct}/{class_total})")

    # Save report to file
    with open('results/model_evaluation_report.txt', 'w') as f:
        f.write("CircuitGuard - Model Evaluation Report\n")
        f.write("="*50 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
        f.write(f"Total Samples: {total}\n")
        f.write(f"Correct Predictions: {correct}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(report)
        f.write("\n\nAccuracy per Defect Type:\n")
        for i, class_name in enumerate(class_names):
            class_mask = np.array(labels) == i
            if np.sum(class_mask) > 0:
                class_correct = np.sum(np.array(predictions)[class_mask] == i)
                class_total = np.sum(class_mask)
                class_acc = 100 * class_correct / class_total
                f.write(
                    f"  {class_name}: {class_acc:.2f}% ({class_correct}/{class_total})\n")


def main():
    print("🔍 Evaluating Trained PCB Defect Classification Model")
    print("="*70)

    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Check which model file exists
    model_paths = [
        'results/best_model.pth',
        'results/pcb_defect_model.pth',
        'pcb_defect_model.pth'
    ]

    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        print("❌ No trained model found. Please train the model first.")
        print("💡 Run: python model_training.py")
        return

    # Data transforms (same as validation during training)
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

    print(f"📁 Dataset Info:")
    print(f"   • Total images: {len(dataset)}")
    print(f"   • Number of classes: {len(dataset.class_names)}")
    print(f"   • Classes: {dataset.class_names}")

    # Use entire dataset for evaluation (or split if you want)
    data_loader = DataLoader(dataset, batch_size=32,
                             shuffle=False, num_workers=2)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_trained_model(model_path, len(dataset.class_names))

    if model is None:
        return

    model = model.to(device)

    # Evaluate model
    predictions, labels, probabilities = evaluate_model(
        model, data_loader, device, dataset.class_names)

    # Generate reports and visualizations
    print_detailed_report(labels, predictions, dataset.class_names)
    plot_confusion_matrix(labels, predictions, dataset.class_names)
    plot_class_accuracy(labels, predictions, dataset.class_names)

    print(f"\n✅ EVALUATION COMPLETED!")
    print(f"📁 Results saved in 'results/' folder:")
    print(f"   • model_evaluation_report.txt - Detailed accuracy report")
    print(f"   • model_evaluation_cm.png - Confusion matrix")
    print(f"   • class_accuracy.png - Accuracy per defect type")


if __name__ == "__main__":
    main()
