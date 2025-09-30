# baseline/inference_engine.py
"""
Core inference engine - handles YOLO model operations
This is the ONLY file that directly interacts with YOLO
"""
import time
from ultralytics import YOLO
from typing import Dict

class InferenceEngine:
    """
    YOLO model operations
    Model loading and inference ONLY
    """
    
    def __init__(self, model_path: str, device: int = 0):
        """Initialize with model path"""
        self.model_path = model_path
        self.device = device
        self.model = None
        
    def load_model(self):
        """Load YOLO model"""
        print(f"Loading model from: {self.model_path}")
        self.model = YOLO(self.model_path)
        print("Model loaded successfully")
        return self.model
    
    def predict_frame(self, frame, img_size: int = 640) -> tuple:
        """
        Run inference on single frame
        Returns: (results, latency_ms)
        """
        if self.model is None:
            self.load_model()
            
        t_start = time.perf_counter()
        results = self.model.predict(
            source=frame,
            imgsz=img_size,
            verbose=False,
            device=self.device
        )
        t_end = time.perf_counter()
        
        latency_ms = (t_end - t_start) * 1000.0
        
        return results, latency_ms
    
    def validate_coco(self, data_yaml: str, img_size: int = 640) -> Dict:
        """
        Run COCO validation
        Returns: metrics dictionary
        """
        if self.model is None:
            self.load_model()
            
        print("Running COCO validation...")
        t0 = time.perf_counter()
        results = self.model.val(
            data=data_yaml,
            imgsz=img_size,
            device=self.device,
            verbose=False
        )
        t1 = time.perf_counter()
        
        # Extract metrics 
        total_time = t1 - t0
        
        return {
            'results': results,
            'total_time_s': total_time,
            'latency_ms': total_time * 1000.0
        }
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        if self.model is None:
            return {'error': 'Model not loaded'}
        
        return {
            'model_path': str(self.model_path),
            'model_type': self.model.__class__.__name__,
        }