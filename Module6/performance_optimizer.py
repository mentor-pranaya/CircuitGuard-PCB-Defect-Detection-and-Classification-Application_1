import torch
import time
from functools import lru_cache
from typing import List
import gc

class PerformanceOptimizer:
    """Performance optimization utilities for the backend"""
    
    def __init__(self):
        self.batch_size = 4  # Optimal for most GPUs
        self.max_image_size = (1024, 1024)  # Prevent memory issues
    
    @staticmethod
    def clear_gpu_memory():
        """Clear GPU memory to prevent leaks"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    @staticmethod
    def optimize_model_for_inference(model: torch.nn.Module):
        """Optimize model for faster inference"""
        model.eval()
        
        # Enable GPU optimizations
        if torch.cuda.is_available():
            model = model.cuda()
            torch.backends.cudnn.benchmark = True
        
        return model
    
    @lru_cache(maxsize=10)
    def get_optimal_batch_size(self, image_size: tuple) -> int:
        """Calculate optimal batch size based on image size"""
        total_pixels = image_size[0] * image_size[1]
        
        if total_pixels > 1000000:  # Large images
            return 1
        elif total_pixels > 500000:  # Medium images
            return 2
        else:  # Small images
            return 4
    
    def preprocess_batch(self, images: List, target_size: tuple = (128, 128)):
        """Optimized batch preprocessing"""
        # Implement batch preprocessing logic
        pass

# Usage example
if __name__ == "__main__":
    # Test the backend
    model_path = r"C:\Users\Kandu\OneDrive\Desktop\info_temp\PCB_DATASET\modules\module3_model_training\checkpoints\best_model.pth"
    
    try:
        detector = PCBDefectDetector(model_path)
        print("✅ Backend initialized successfully!")
        print(f"🔧 System info: {detector.get_system_info()}")
    except Exception as e:
        print(f"❌ Backend initialization failed: {e}")