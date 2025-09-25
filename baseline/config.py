# baseline/config.py
from dataclasses import dataclass, asdict
import yaml

@dataclass
class Config:
    """Configuration for baseline inference"""
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
    FPS_ALPHA: float = 0.1
    
    @classmethod
    def from_yaml(cls, path: str):
        """Load config from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_dict(self):
        """Convert to dictionary"""
        return asdict(self)