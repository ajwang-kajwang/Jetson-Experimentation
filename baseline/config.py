# ============================================================================
# CONFIGURATION
# ============================================================================
from dataclasses import dataclass

@dataclass
class Config:
    """Central configuration for all experiments"""
    # Model settings
    MODEL_PT: str = "yolov8s.pt"
    IMG_SIZE: int = 640
    
    # Hardware settings
    CAMERA_INDEX: int = 0
    DEVICE: int = 0  # CUDA device
    
    # Dataset settings
    DATA_YAML: str = "coco.yaml"
    
    # Metrics settings
    WARMUP_FRAMES: int = 20
    METRIC_BUFFER_SIZE: int = 100
    FPS_ALPHA: float = 0.1  # Smoothing factor for FPS