"""
CircuitGuard - PCB Defect Detection Backend
Filename: backend_inference.py
"""
import torch
import torch.nn as nn
from torchvision import transforms
import timm
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import os
from typing import Dict, Tuple, Optional

class InferenceConfig:
    IMG_SIZE = 128
    NUM_CLASSES = 6
    MODEL_NAME = 'efficientnet_b4' 
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BBOX_COLOR = (59, 130, 246)
    BBOX_WIDTH = 4
    TEXT_COLOR = (255, 255, 255)
    TEXT_BG_COLOR = (15, 23, 42)
    FONT_SIZE = 24

def load_model_and_mapping(
    model_path='best_efficientnet_b4.pth', 
    class_mapping_path='class_mapping.json', 
    device=None
):
    if device is None: device = InferenceConfig.DEVICE
    
    if not os.path.exists(class_mapping_path): raise FileNotFoundError("Missing class_mapping.json")
    with open(class_mapping_path, 'r') as f: class_mapping = json.load(f)
    
    model = timm.create_model(InferenceConfig.MODEL_NAME, pretrained=False, num_classes=InferenceConfig.NUM_CLASSES)
    
    if not os.path.exists(model_path): raise FileNotFoundError(f"Missing model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint
        
    model = model.to(device).eval()
    return model, class_mapping

def predict_defect(model, image, class_mapping, device=None):
    if device is None: device = InferenceConfig.DEVICE
    tf = transforms.Compose([
        transforms.Resize((InferenceConfig.IMG_SIZE, InferenceConfig.IMG_SIZE)),
        transforms.ToTensor()
    ])
    
    if image.mode != 'RGB': image = image.convert('RGB')
    
    tensor = tf(image) * 255.0  
    tensor = tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)
        conf, idx = torch.max(probs, 1)
        
    idx = idx.item()
    cls_name = class_mapping['idx_to_class'][str(idx)]
    
    return {
        'predicted_class': cls_name,
        'confidence': conf.item()
    }

def visualize_prediction(image, prediction):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    try:
        font = ImageFont.truetype("arial.ttf", InferenceConfig.FONT_SIZE)
    except:
        font = ImageFont.load_default()
    draw.rectangle([10, 10, w-10, h-10], outline=InferenceConfig.BBOX_COLOR, width=InferenceConfig.BBOX_WIDTH)
    return img
