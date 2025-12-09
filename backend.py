import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import base64
from io import BytesIO
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class PCBDefectDetector:
    def __init__(self, model_path: str = 'efficientnet_b4_model.pth'):
        """Initialize PCB defect detector with trained model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        
        # Defect class mapping
        self.defect_classes = [
            'missing_hole', 'mouse_bite', 'open_circuit',
            'short', 'spur', 'spurious_copper'
        ]
        
        # Color mapping for visualization
        self.defect_colors = {
            'missing_hole': (239, 68, 68),      # Red
            'mouse_bite': (245, 158, 11),      # Yellow
            'open_circuit': (16, 185, 129),    # Green
            'short': (59, 130, 246),           # Blue
            'spur': (139, 92, 246),            # Purple
            'spurious_copper': (236, 72, 153)  # Pink
        }
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained EfficientNet model"""
        model = models.efficientnet_b4(pretrained=False)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 6)  # 6 defect classes
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found. Using untrained model.")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _get_transform(self) -> transforms.Compose:
        """Get image transformation pipeline"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess_images(self, template_path: str, test_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Align and subtract template from test image"""
        try:
            # Read images
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            test = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
            
            if template is None or test is None:
                raise ValueError("Could not read images")
            
            # Align images using ORB feature matching
            aligned_test = self._align_images(template, test)
            
            # Subtract template from test
            diff = cv2.absdiff(template, aligned_test)
            
            # Apply thresholding
            _, binary_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations
            kernel = np.ones((3, 3), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            
            return binary_mask, aligned_test
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None, None
    
    def _align_images(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Align two images using ORB feature matching"""
        # Initialize ORB detector
        orb = cv2.ORB_create(1000)
        
        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        # Create BFMatcher and match descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Get matching points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
        
        # Find homography
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        # Warp image
        height, width = img1.shape
        aligned = cv2.warpPerspective(img2, M, (width, height))
        
        return aligned
    
    def detect_defects(self, mask: np.ndarray, original_image: np.ndarray) -> List[Dict]:
        """Detect defects from binary mask and extract regions"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        defects = []
        min_area = 50
        max_area = 5000
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Add padding to ROI
                padding = 10
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(original_image.shape[1], x + w + padding)
                y2 = min(original_image.shape[0], y + h + padding)
                
                # Crop defect region
                defect_roi = original_image[y1:y2, x1:x2]
                
                # Classify defect
                defect_type, confidence = self.classify_defect(defect_roi)
                
                defects.append({
                    'id': i + 1,
                    'type': defect_type,
                    'confidence': float(confidence),
                    'bbox': [int(x), int(y), int(x + w), int(y + h)],
                    'area': int(area)
                })
        
        return defects
    
    def classify_defect(self, defect_image: np.ndarray) -> Tuple[str, float]:
        """Classify defect using trained model"""
        try:
            # Convert to PIL Image
            if len(defect_image.shape) == 2:
                defect_image = cv2.cvtColor(defect_image, cv2.COLOR_GRAY2RGB)
            pil_image = Image.fromarray(defect_image)
            
            # Apply transformations
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            defect_type = self.defect_classes[predicted.item()]
            confidence_score = confidence.item()
            
            return defect_type, confidence_score
            
        except Exception as e:
            print(f"Error in classification: {e}")
            return 'unknown', 0.0
    
    def annotate_image(self, image: np.ndarray, defects: List[Dict]) -> np.ndarray:
        """Annotate image with defect bounding boxes and labels"""
        if len(image.shape) == 2:
            annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            annotated = image.copy()
        
        for defect in defects:
            x1, y1, x2, y2 = defect['bbox']
            defect_type = defect['type']
            confidence = defect['confidence']
            
            # Get color for this defect type
            color = self.defect_colors.get(defect_type, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{defect_type}: {confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return annotated
    
    def process_request(self, template_data: bytes, test_data: bytes) -> Dict:
        """Process complete defect detection pipeline"""
        try:
            # Save uploaded images temporarily
            template_path = "temp_template.jpg"
            test_path = "temp_test.jpg"
            
            with open(template_path, 'wb') as f:
                f.write(template_data)
            with open(test_path, 'wb') as f:
                f.write(test_data)
            
            # Preprocess images
            mask, aligned_test = self.preprocess_images(template_path, test_path)
            
            if mask is None:
                return {"error": "Failed to preprocess images"}
            
            # Detect and classify defects
            defects = self.detect_defects(mask, aligned_test)
            
            # Load original test image for annotation
            test_image = cv2.imread(test_path, cv2.IMREAD_COLOR)
            
            # Annotate image
            annotated_image = self.annotate_image(test_image, defects)
            
            # Convert images to base64 for frontend display
            _, annotated_buffer = cv2.imencode('.jpg', annotated_image)
            _, mask_buffer = cv2.imencode('.jpg', mask)
            
            annotated_b64 = base64.b64encode(annotated_buffer).decode('utf-8')
            mask_b64 = base64.b64encode(mask_buffer).decode('utf-8')
            
            # Clean up temp files
            os.remove(template_path)
            os.remove(test_path)
            
            return {
                "success": True,
                "defects": defects,
                "annotated_image": annotated_b64,
                "mask_image": mask_b64,
                "total_defects": len(defects),
                "defect_types": [d['type'] for d in defects]
            }
            
        except Exception as e:
            return {"error": str(e)}
