# baseline/baseline_infer_pytorch.py
"""
Main baseline inference script
Coordinates all other components
"""
import time
import numpy as np
import cv2
import argparse
import json
from pathlib import Path

# Import configuration 
from config import (
    MODEL_PT, IMG_SIZE, CAMERA_INDEX, DATA_YAML,
    EnhancedConfig  
)

# Import modules
from inference_engine import InferenceEngine
from metrics_collector import MetricsCollector
from visualizer import Visualizer
from performance_analyzer import PerformanceAnalyzer

def infer_live(camera_index, metrics_out='baseline_live_metrics.json'):
    """
    Live inference function 
    """
    # Initialize components
    engine = InferenceEngine(MODEL_PT)
    collector = MetricsCollector(warmup_frames=0)  
    visualizer = Visualizer()
    
    # Load model
    engine.load_model()
    
    # Open camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError('Camera not available')
    
    print(f"Starting live inference on camera {camera_index}")
    print("Press ESC to stop...")
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            # Run inference
            results, latency_ms = engine.predict_frame(frame, IMG_SIZE)
            
            # Collect metrics
            stats = collector.add_frame_metrics(latency_ms)
            
            # Get annotated frame
            annotated = results[0].plot()
            
            # Add overlay (original style)
            text = f"Latency: {latency_ms:.1f} ms  FPS: {stats['fps_smooth']:.1f}"
            cv2.putText(annotated, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Display
            cv2.imshow('PyTorch YOLOv8x (Baseline)', annotated)
            
            # Check for ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Get final metrics
        metrics = collector.get_final_metrics()
        
        # Save metrics
        if metrics:
            print("\nFinal Metrics:")
            print(metrics)
            with open(metrics_out, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved live metrics to {metrics_out}")

def infer_coco(metrics_out='baseline_coco_metrics.json'):
    """
    COCO validation
    """
    engine = InferenceEngine(MODEL_PT)
    
    # Run validation
    val_results = engine.validate_coco(DATA_YAML, IMG_SIZE)
    
    # Extract metrics 
    results = val_results['results']
    
    def to_scalar(x):
        a = np.asarray(x)
        if a.size == 0:
            return float('nan')
        if a.shape == ():
            return float(a)
        return float(np.nanmean(a))
    
    try:
        metrics = {
            'precision': to_scalar(results.box.p),
            'recall': to_scalar(results.box.r),
            'map50': to_scalar(results.box.map50),
            'map50_95': to_scalar(results.box.map),
            'mean_latency_ms': float(val_results['latency_ms']),
            'p95_latency_ms': float(val_results['latency_ms']),
            'fps': 1000.0 / val_results['latency_ms']
        }
    except AttributeError:
        metrics = {'error': 'Could not extract metrics'}
    
    print(metrics)
    with open(metrics_out, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved COCO metrics to {metrics_out}")

def infer_enhanced(mode='live', camera_index=None, metrics_out=None):
    """
    Enhanced inference with detailed metrics and analysis
    Optional: Use --enhanced flag to access this
    """
    config = EnhancedConfig()
    
    # Initialize components with enhanced settings
    engine = InferenceEngine(config.model_pt, config.device)
    collector = MetricsCollector(config.warmup_frames, config.metric_buffer_size)
    visualizer = Visualizer(show_enhanced=True)
    analyzer = PerformanceAnalyzer()
    
    if mode == 'live':
        # Enhanced live inference with all features
        # ... implementation with warmup, enhanced display, etc.
        pass
    
    # Get metrics
    metrics = collector.get_final_metrics()
    
    # Analyze
    insights = analyzer.analyze_metrics(metrics)
    
    # Save enhanced output
    output = {
        'config': config.to_dict(),
        'metrics': metrics,
        'analysis': insights,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    output_file = metrics_out or f'enhanced_{mode}_metrics.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    return metrics

# Main entry point 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['live', 'coco'], default='coco')
    parser.add_argument('--camera', type=int, default=CAMERA_INDEX)
    parser.add_argument('--metrics_out', type=str, default=None)
    parser.add_argument('--enhanced', action='store_true', 
                       help='Use enhanced metrics and analysis')
    args = parser.parse_args()
    
    if args.enhanced:
        # Enhanced mode with full features
        infer_enhanced(args.mode, args.camera, args.metrics_out)
    else:
        # Original mode - exactly as expected
        if args.mode == 'live':
            out = args.metrics_out or 'baseline_live_metrics.json'
            infer_live(args.camera, metrics_out=out)
        else:
            out = args.metrics_out or 'baseline_coco_metrics.json'
            infer_coco(metrics_out=out)