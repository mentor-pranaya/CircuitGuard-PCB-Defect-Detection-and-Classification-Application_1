import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image, ImageDraw
import cv2
import numpy as np
import os
import time
import logging
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Data class to store detection results"""
    defect_type: str
    confidence: float
    defect_regions: List[Tuple[int, int, int]]  # (x, y, radius)
    processing_time: float
    all_probabilities: Dict[str, float]
    annotated_image: np.ndarray
    original_size: Tuple[int, int]

class PCBDefectDetector:
    """Modular backend engine for PCB defect detection"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the defect detection engine
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on ('auto', 'cuda', 'cpu')
        """
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self.transform = None
        self.is_initialized = False
        
        # Defect configuration
        self.CLASS_NAMES = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
        self.DEFECT_COLORS = {
            'Missing_hole': (255, 0, 0),      # Red
            'Mouse_bite': (255, 165, 0),      # Orange
            'Open_circuit': (0, 0, 255),      # Blue
            'Short': (128, 0, 128),           # Purple
            'Spur': (255, 255, 0),            # Yellow
            'Spurious_copper': (0, 255, 255)  # Cyan
        }
        
        self._initialize_model()
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == "auto":
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _initialize_model(self):
        """Initialize model and preprocessing transforms"""
        try:
            # Load model
            self.model = EfficientNetB4Classifier(num_classes=len(self.CLASS_NAMES))
            
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                else:
                    self.model.load_state_dict(checkpoint)
                
                logger.info(f"✅ Model loaded successfully from {self.model_path}")
            else:
                logger.warning("⚠️ Model file not found. Using pretrained weights.")
            
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize transforms
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            self.is_initialized = True
            logger.info(f"🚀 Backend engine initialized on device: {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize model: {e}")
            raise
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for model inference
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed tensor
        """
        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call _initialize_model first.")
        
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def predict_defect(self, input_tensor: torch.Tensor) -> Tuple[str, float, Dict[str, float]]:
        """
        Run defect classification inference
        
        Args:
            input_tensor: Preprocessed image tensor
            
        Returns:
            Tuple of (defect_type, confidence, all_probabilities)
        """
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Convert to Python types
        defect_type = self.CLASS_NAMES[predicted_idx.item()]
        confidence_score = confidence.item()
        
        # Create probability dictionary
        all_probs = {
            cls: float(prob) 
            for cls, prob in zip(self.CLASS_NAMES, probabilities[0].cpu().numpy())
        }
        
        inference_time = time.time() - start_time
        logger.info(f"🔍 Inference completed in {inference_time:.3f}s - {defect_type} ({confidence_score:.3f})")
        
        return defect_type, confidence_score, all_probs
    
    def detect_defect_regions(self, image: np.ndarray, defect_type: str) -> List[Tuple[int, int, int]]:
        """
        Detect defect regions in the image (simulated for now)
        
        Args:
            image: Input image as numpy array
            defect_type: Type of defect to look for
            
        Returns:
            List of (x, y, radius) tuples for defect regions
        """
        height, width = image.shape[:2]
        regions = []
        
        # Simulate defect region detection based on defect type
        # In production, this would use object detection or segmentation models
        if defect_type == 'Missing_hole':
            regions = [(width//4, height//4, 20), (3*width//4, height//4, 15)]
        elif defect_type == 'Mouse_bite':
            regions = [(width//3, height//2, 25), (2*width//3, height//3, 20)]
        elif defect_type == 'Open_circuit':
            regions = [(width//2, height//3, 30)]
        elif defect_type == 'Short':
            regions = [(width//2, height//2, 25)]
        elif defect_type == 'Spur':
            regions = [(width//5, 4*height//5, 20), (4*width//5, height//5, 15)]
        elif defect_type == 'Spurious_copper':
            regions = [(width//6, height//6, 35), (5*width//6, 5*height//6, 30)]
        
        logger.info(f"📍 Detected {len(regions)} defect regions for {defect_type}")
        return regions
    
    def create_annotated_image(self, original_image: np.ndarray, defect_type: str, 
                             confidence: float, regions: List[Tuple[int, int, int]]) -> np.ndarray:
        """
        Create annotated image with defect highlights
        
        Args:
            original_image: Original image as numpy array
            defect_type: Type of defect detected
            confidence: Confidence score
            regions: List of defect regions
            
        Returns:
            Annotated image as numpy array
        """
        # Create copy for annotation
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            annotated = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        else:
            annotated = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        
        # Resize for better visualization
        height, width = annotated.shape[:2]
        if max(height, width) > 800:
            scale = 800 / max(height, width)
            new_size = (int(width * scale), int(height * scale))
            annotated = cv2.resize(annotated, new_size)
            # Scale regions accordingly
            regions = [(int(x*scale), int(y*scale), int(r*scale)) for x, y, r in regions]
        
        # Get defect color
        defect_color = self.DEFECT_COLORS.get(defect_type, (255, 0, 0))
        
        # Highlight defect regions
        for i, (x, y, radius) in enumerate(regions):
            # Draw highlighted circle
            cv2.circle(annotated, (x, y), radius, defect_color, 3)
            cv2.circle(annotated, (x, y), radius, defect_color, -1)
            
            # Add defect number
            cv2.putText(annotated, f"D{i+1}", (x-10, y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add information overlay
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
        
        # Add text information
        cv2.putText(annotated, "DEFECT DETECTED", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated, f"Type: {defect_type}", (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, defect_color, 2)
        cv2.putText(annotated, f"Confidence: {confidence:.1%}", (20, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, defect_color, 2)
        
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    def process_image(self, image: Image.Image) -> DetectionResult:
        """
        Complete pipeline for processing a single image
        
        Args:
            image: PIL Image object to process
            
        Returns:
            DetectionResult object with all results
        """
        start_time = time.time()
        
        if not self.is_initialized:
            raise RuntimeError("Backend engine not initialized")
        
        try:
            # Store original size
            original_size = image.size
            
            # Convert to numpy for processing
            image_np = np.array(image)
            
            # Step 1: Preprocess
            input_tensor = self.preprocess_image(image)
            
            # Step 2: Predict defect
            defect_type, confidence, all_probs = self.predict_defect(input_tensor)
            
            # Step 3: Detect regions
            defect_regions = self.detect_defect_regions(image_np, defect_type)
            
            # Step 4: Create annotated image
            annotated_image = self.create_annotated_image(
                image_np, defect_type, confidence, defect_regions
            )
            
            processing_time = time.time() - start_time
            
            logger.info(f"✅ Processing completed in {processing_time:.3f}s")
            
            return DetectionResult(
                defect_type=defect_type,
                confidence=confidence,
                defect_regions=defect_regions,
                processing_time=processing_time,
                all_probabilities=all_probs,
                annotated_image=annotated_image,
                original_size=original_size
            )
            
        except Exception as e:
            logger.error(f"❌ Error processing image: {e}")
            raise
    
    def batch_process(self, images: List[Image.Image]) -> List[DetectionResult]:
        """
        Process multiple images in batch (optimized)
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            List of DetectionResult objects
        """
        results = []
        for i, image in enumerate(images):
            logger.info(f"Processing image {i+1}/{len(images)}")
            try:
                result = self.process_image(image)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process image {i+1}: {e}")
                # Continue with next image
                continue
        
        return results
    
    def get_system_info(self) -> Dict:
        """Get system information and model stats"""
        return {
            "device": str(self.device),
            "model_initialized": self.is_initialized,
            "defect_classes": self.CLASS_NAMES,
            "model_path": self.model_path,
            "cuda_available": torch.cuda.is_available()
        }

# Model class (same as training)
class EfficientNetB4Classifier(nn.Module):
    def __init__(self, num_classes=6):
        super(EfficientNetB4Classifier, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)