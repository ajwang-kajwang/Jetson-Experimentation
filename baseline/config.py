# baseline/config.py
"""
Enhanced settings
"""
from pathlib import Path
from dataclasses import dataclass

# ============================================================================
# ORIGINAL CONFIGURATION
# ============================================================================
LAB_DIR = Path.cwd()  # or Path('/workspace') inside container
LAB_DIR.mkdir(exist_ok=True, parents=True)
BASE_DIR = Path(__file__).parent

# Model and I/O
MODEL_PT = str(LAB_DIR / 'yolov8x.pt')  
IMG_SIZE = 640
BATCH = 1   
DYNAMIC = False
HALF = False

# Artifacts
ONNX_PATH = LAB_DIR / 'yolov8x.onnx'
ENGINE_PATH = LAB_DIR / 'yolov8x_trt_from_onnx.engine'
ONNX_PATH_FP16 = LAB_DIR / 'yolov8x_fp16.onnx'
ENGINE_PATH_FP16 = LAB_DIR / 'yolov8x_trt_from_onnx_fp16.engine'

ENGINE_PATH_T2T = LAB_DIR / 'yolov8x_trt_from_torch2trt.engine'
ENGINE_PATH_T2T_FP16 = LAB_DIR / 'yolov8x_trt_from_torch2trt_fp16.engine'
T2TRT_STATE = LAB_DIR / 'yolov8x_torch2trt.pth'
T2TRT_STATE_FP16 = LAB_DIR / 'yolov8x_torch2trt_fp16.pth'

# Dataset
DATA_YAML = str(BASE_DIR / 'datasets' / 'coco' / 'coco.yaml')

# Camera
CAMERA_INDEX = 0  # replace with GStreamer string on Jetson if needed

# ============================================================================
# Additional settings for better metrics
# ============================================================================
@dataclass
class EnhancedConfig:
    """Enhanced configuration for detailed metrics and analysis"""
    # Use original values as defaults
    model_pt: str = MODEL_PT
    img_size: int = IMG_SIZE
    camera_index: int = CAMERA_INDEX
    data_yaml: str = DATA_YAML
    batch_size: int = BATCH
    
    # Additional metrics settings
    warmup_frames: int = 20          # Frames to skip before collecting metrics
    metric_buffer_size: int = 100    # Rolling buffer for real-time stats
    fps_alpha: float = 0.1           # FPS smoothing factor
    
    # Device settings
    device: int = 0                  # CUDA device index
    
    # Display settings
    show_enhanced_metrics: bool = True
    save_video: bool = False
    
    # Analysis thresholds
    latency_threshold_ms: float = 33.0  # For 30 FPS
    stability_ratio_threshold: float = 1.5
    
    def to_dict(self):
        """Convert to dictionary for JSON export"""
        return {
            'model_pt': str(self.model_pt),
            'img_size': self.img_size,
            'camera_index': self.camera_index,
            'warmup_frames': self.warmup_frames,
            'device': self.device
        }