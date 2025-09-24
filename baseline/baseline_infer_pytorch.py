# baseline_infer_pytorch.py
"""
Task 0: Baseline PyTorch Live Inference
Refactored version with clear class structure
"""
import time
import numpy as np
import cv2
import argparse
import json
from collections import deque
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from ultralytics import YOLO


# ============================================================================
# INFERENCE ENGINES
# ============================================================================
class BaselineInferenceEngine:
    """Main inference engine for baseline testing"""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.model = None
        self.metrics_collector = MetricsCollector(
            warmup_frames=config.WARMUP_FRAMES,
            buffer_size=config.METRIC_BUFFER_SIZE
        )
        self.visualizer = Visualizer()
        self.analyzer = PerformanceAnalyzer()
    
    def load_model(self) -> None:
        """Load YOLO model"""
        print(f"Loading model: {self.config.MODEL_PT}")
        self.model = YOLO(self.config.MODEL_PT)
        print("Model loaded successfully")
    
    def print_system_info(self) -> None:
        """Print system information"""
        import torch
        print("=" * 60)
        print("SYSTEM INFORMATION")
        print("=" * 60)
        print(f"Model: {self.config.MODEL_PT}")
        print(f"Image Size: {self.config.IMG_SIZE}x{self.config.IMG_SIZE}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("=" * 60)
    
    def run_live_inference(self, camera_index: int = None, 
                          display: bool = True) -> Dict[str, float]:
        """Run live inference on webcam"""
        if camera_index is None:
            camera_index = self.config.CAMERA_INDEX
        
        # Initialize
        self.print_system_info()
        self.load_model()
        
        # Open camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Camera {camera_index} not available")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.IMG_SIZE)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.IMG_SIZE)
        
        print(f"\nStarting inference... Press ESC to stop")
        print(f"Warming up for {self.config.WARMUP_FRAMES} frames...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run inference
                t_start = time.perf_counter()
                results = self.model.predict(
                    source=frame,
                    imgsz=self.config.IMG_SIZE,
                    verbose=False,
                    device=self.config.DEVICE
                )
                t_end = time.perf_counter()
                
                # Calculate latency
                latency_ms = (t_end - t_start) * 1000.0
                
                # Update metrics
                self.metrics_collector.add_measurement(latency_ms, t_end)
                
                # Get current stats
                stats = self.metrics_collector.get_current_stats()
                stats['current_latency'] = latency_ms
                
                # Display
                if display:
                    # Get annotated frame with detections
                    annotated = results[0].plot()
                    
                    # Add metrics overlay
                    annotated = self.visualizer.draw_metrics(
                        annotated, stats,
                        warming_up=self.metrics_collector.is_warming_up()
                    )
                    
                    # Show and check for exit
                    if not self.visualizer.show(annotated):
                        break
                
                # Periodic console output
                if self.metrics_collector.frame_count % 100 == 0:
                    self._print_periodic_stats(stats)
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Calculate final metrics
            metrics = self.metrics_collector.calculate_final_metrics()
            return metrics
    
    def _print_periodic_stats(self, stats: Dict[str, float]) -> None:
        """Print periodic statistics to console"""
        if not self.metrics_collector.is_warming_up() and 'mean_latency' in stats:
            print(f"\nFrame {stats['frame_count']}: "
                  f"Mean={stats['mean_latency']:.1f}ms, "
                  f"P95={stats['p95_latency']:.1f}ms, "
                  f"FPS={stats['fps_avg']:.1f}")
    
    def run_coco_validation(self) -> Dict[str, float]:
        """Run COCO validation"""
        self.print_system_info()
        self.load_model()
        
        print("\nRunning COCO validation...")
        
        t0 = time.perf_counter()
        results = self.model.val(
            data=self.config.DATA_YAML,
            imgsz=self.config.IMG_SIZE,
            device=self.config.DEVICE,
            verbose=False
        )
        t1 = time.perf_counter()
        
        # Extract metrics safely
        def to_scalar(x):
            a = np.asarray(x)
            return float(np.nanmean(a)) if a.size > 0 else float('nan')
        
        try:
            metrics = {
                'precision': to_scalar(results.box.p),
                'recall': to_scalar(results.box.r),
                'map50': to_scalar(results.box.map50),
                'map50_95': to_scalar(results.box.map),
                'total_time_s': float(t1 - t0),
                'mean_latency_ms': float((t1 - t0) * 1000.0)
            }
        except AttributeError:
            # Fallback for older versions
            metrics = {'error': 'Could not extract metrics'}
        
        return metrics
    
    def save_metrics(self, metrics: Dict, filename: str) -> None:
        """Save metrics to JSON file"""
        if not metrics:
            print("No metrics to save")
            return
        
        # Add metadata
        output = {
            'config': asdict(self.config),
            'metrics': metrics,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add analysis insights
        insights = self.analyzer.analyze(metrics)
        if insights:
            output['analysis'] = insights
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nMetrics saved to {filename}")
        
        # Print summary
        self._print_final_summary(metrics, insights)
    
    def _print_final_summary(self, metrics: Dict, insights: Dict) -> None:
        """Print final summary of metrics and insights"""
        print("\n" + "=" * 60)
        print("FINAL METRICS SUMMARY")
        print("=" * 60)
        
        if 'frames_processed' in metrics:
            print(f"Frames Processed: {metrics['frames_processed']}")
            print(f"Total Time: {metrics['total_time_s']:.2f}s")
        
        print("\nLatency Statistics:")
        for key in ['mean', 'p50', 'p95', 'p99', 'min', 'max']:
            metric_key = f"{key}_latency_ms"
            if metric_key in metrics:
                print(f"  {key.upper()}: {metrics[metric_key]:.2f} ms")
        
        if 'effective_fps' in metrics:
            print(f"\nThroughput: {metrics['effective_fps']:.2f} FPS")
        
        if insights:
            print("\nAnalysis:")
            for key, value in insights.items():
                print(f"  {key}: {value}")
        
        print("=" * 60)

