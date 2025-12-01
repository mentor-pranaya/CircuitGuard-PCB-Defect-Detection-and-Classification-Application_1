import torch
import torch.nn as nn
from torchvision import transforms
import timm
from PIL import Image
import os
import glob
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
MODEL_CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, 'efficientnet_b4_pcb_classifier_best.pth')
MODEL_NAME = 'efficientnet_b4' 
IMAGE_SIZE = 128       
INFERENCE_ROOT = 'C:\\Users\\Lenovo\\PCB Python\\validation_set' 

CLASS_NAMES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spurious_copper', 'spur'] 

# Standard normalization parameters
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def load_inference_model(num_classes, model_name, checkpoint_path):
    
    model = timm.create_model(model_name, pretrained=False) 
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Ensure Module 3 training ran successfully.")
    
    device = torch.device("cpu") 
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()  # Set model to evaluation mode 
    
    print(f"Model loaded successfully from {checkpoint_path} onto {device}. Ready for inference.")
    return model, device

inference_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD)])


def predict_single_image(model, device, image_path, class_names):
#for single image
    
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return "Prediction Error", 0.0
    input_tensor = inference_transforms(image).unsqueeze(0) 
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)

    probabilities = torch.softmax(output, dim=1)
    confidence, predicted_index = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted_index.item()]
    confidence_score = confidence.item() * 100
    
    return predicted_class, confidence_score


if __name__ == "__main__":
    if INFERENCE_ROOT.startswith('path/to/'):
        print("!!! ACTION REQUIRED: UPDATE INFERENCE_ROOT PATH WITH YOUR DATA !!!")
    else:
        num_classes = len(CLASS_NAMES)
        model, device = load_inference_model(num_classes, MODEL_NAME, MODEL_CHECKPOINT_PATH)
        all_test_files = []
        for ext in ('*.jpg', '*.png'):
            all_test_files.extend(glob.glob(os.path.join(INFERENCE_ROOT, ext), recursive=False))
        
        if not all_test_files:
            print(f"No images found in the inference path: {INFERENCE_ROOT}")
        else:
            print(f"Found {len(all_test_files)} images for inference. Starting prediction...")
            print("\n--- INFERENCE RESULTS ---")
            for i, image_path in enumerate(all_test_files):
                predicted_class, confidence = predict_single_image(model, device, image_path, CLASS_NAMES)
                filename = os.path.basename(image_path)
                
                print(f"[{i+1}/{len(all_test_files)}] {filename.ljust(30)} -> PREDICTED: {predicted_class.ljust(20)} CONFIDENCE: {confidence:.2f}%")
            
            print("\n--- INFERENCE COMPLETE ---")