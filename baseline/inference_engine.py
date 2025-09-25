class BaselineInferenceEngine:
    """
    Enhanced wrapper that uses the modular components
    while maintaining compatibility with original functions
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.model = None
        self.collector = MetricsCollector(self.config.WARMUP_FRAMES)
        self.visualizer = Visualizer()
        self.analyzer = PerformanceAnalyzer()
        
    def run_live_inference(self, camera_index: Optional[int] = None,
                          metrics_out: str = 'baseline_live_metrics.json') -> Dict:
        """Run live inference using class structure"""
        camera_index = camera_index or self.config.CAMERA_INDEX
        
        # Use the original function with classes enabled
        infer_live(camera_index, metrics_out, use_classes=True)
        
        # Load and return metrics
        with open(metrics_out, 'r') as f:
            return json.load(f)
    
    def run_coco_validation(self, metrics_out: str = 'baseline_coco_metrics.json') -> Dict:
        """Run COCO validation"""
        return infer_coco(metrics_out)
    
    def print_system_info(self):
        """Print system information"""
        import torch
        print("=" * 60)
        print("SYSTEM INFORMATION")
        print("=" * 60)
        print(f"Model: {self.config.MODEL_PT}")
        print(f"Image Size: {self.config.IMG_SIZE}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("=" * 60)
