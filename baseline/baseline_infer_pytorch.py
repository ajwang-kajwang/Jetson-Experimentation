# baseline_infer_pytorch.py
"""
Task 0: Baseline PyTorch Live Inference
Refactored version with clear class structure
"""
import time
import numpy as np
import cv2
import json
from ultralytics import YOLO
from config import Config
from metrics import MetricsCollector
from visualizer import Visualizer  
from performance_analyzer import PerformanceAnalyzer


# ============================================================================
# MAIN FUNCTIONS - Direct match to original structure
# ============================================================================

def infer_live(camera_index, metrics_out='baseline_live_metrics.json',
               use_classes=True):
    """
    Live inference function matching original signature
    
    Args:
        camera_index: Camera device index
        metrics_out: Output JSON file path
        use_classes: Whether to use class-based structure (for learning)
    """
    # Load configuration
    if use_classes:
        config = Config()
        MODEL_PT = config.MODEL_PT
        IMG_SIZE = config.IMG_SIZE
    else:
        # Direct imports for original compatibility
        from config import MODEL_PT, IMG_SIZE
    
    # Initialize model
    model = YOLO(MODEL_PT)
    
    # Open camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError('Camera not available')
    
    # Initialize metrics collection
    if use_classes:
        collector = MetricsCollector()
        visualizer = Visualizer()
    else:
        # Original style - inline variables
        latencies = []
        t_last = time.perf_counter()
        fps_smooth = 0.0
        alpha = 0.1
    
    print(f"Starting inference on camera {camera_index}")
    print("Press ESC to stop...")
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            # Inference timing
            t0 = time.perf_counter()
            results = model.predict(source=frame, imgsz=IMG_SIZE, 
                                   verbose=False, device=0)
            t1 = time.perf_counter()
            
            # Calculate latency
            lat_ms = (t1 - t0) * 1000.0
            
            if use_classes:
                # Class-based approach
                stats = collector.add_measurement(lat_ms, t1)
                annotated = results[0].plot()
                annotated = visualizer.draw_metrics(
                    annotated, stats['latency_ms'], stats['fps_smooth']
                )
            else:
                # Original approach
                latencies.append(lat_ms)
                annotated = results[0].plot()
                
                # FPS calculation (original style)
                dt = t1 - t_last
                inst_fps = 1.0 / max(dt, 1e-6)
                fps_smooth = ((1 - alpha) * fps_smooth + alpha * inst_fps 
                             if fps_smooth > 0 else inst_fps)
                t_last = t1
                
                # Draw text (original style)
                cv2.putText(annotated, 
                           f'Latency: {lat_ms:.1f} ms  FPS: {fps_smooth:.1f}',
                           (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.9, 
                           (0, 255, 0), 
                           2, 
                           cv2.LINE_AA)
            
            # Display
            cv2.imshow('PyTorch YOLOv8 (Baseline)', annotated)
            
            # Check for ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Save metrics
        if use_classes:
            metrics = collector.get_final_metrics()
        else:
            # Original metric calculation
            if latencies:
                metrics = {
                    'mean_latency_ms': float(np.mean(latencies)),
                    'p95_latency_ms': float(np.percentile(latencies, 95)),
                    'frames': len(latencies),
                    'fps': len(latencies) / (np.sum(latencies) / 1000.0)
                }
            else:
                metrics = {}
        
        if metrics:
            print(metrics)
            with open(metrics_out, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved live metrics to {metrics_out}")
            
            # Optional: Add analysis
            if use_classes:
                analyzer = PerformanceAnalyzer()
                insights = analyzer.analyze(metrics)
                if insights:
                    print("\nAnalysis:", insights)


def infer_coco(metrics_out='baseline_coco_metrics.json'):
    """
    COCO validation function matching original signature exactly
    """
    from config import MODEL_PT, IMG_SIZE, DATA_YAML
    
    model = YOLO(MODEL_PT)
    
    print("Running COCO validation...")
    t0 = time.perf_counter()
    results = model.val(data=DATA_YAML, imgsz=IMG_SIZE, device=0, verbose=False)
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0
    
    # Helper function from original
    def to_scalar(x):
        a = np.asarray(x)
        if a.size == 0:
            return float('nan')
        if a.shape == ():
            return float(a)
        return float(np.nanmean(a))
    
    # Extract metrics (original approach)
    try:
        precision = to_scalar(results.box.p)
        recall = to_scalar(results.box.r)
        f1 = to_scalar(getattr(results.box, 'f1', 
                               np.array([2 * precision * recall / 
                                        max(precision + recall, 1e-12)])))
        map50 = to_scalar(results.box.map50)
        map50_95 = to_scalar(results.box.map)
    except AttributeError:
        # Fallback for older APIs
        p, r, map50_val, map50_95_val = results.mean_results()
        precision = float(p)
        recall = float(r)
        map50 = float(map50_val)
        map50_95 = float(map50_95_val)
        f1 = float(2 * precision * recall / max(precision + recall, 1e-12))
    
    # FPS calculation
    try:
        inf_ms = float(results.speed.get('inference', 0.0))
        fps = 1000.0 / inf_ms if inf_ms > 0 else 1.0 / max((t1 - t0), 1e-6)
    except Exception:
        fps = 1.0 / max((t1 - t0), 1e-6)
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'map50': map50,
        'map50_95': map50_95,
        'mean_latency_ms': float(latency_ms),
        'p95_latency_ms': float(latency_ms),  # single pass
        'fps': float(fps),
    }
    
    print(metrics)
    with open(metrics_out, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved COCO metrics to {metrics_out}")
    
    return metrics