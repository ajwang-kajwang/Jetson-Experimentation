# baseline/metrics_collector.py
"""
Metrics collection and calculation
"""
import time
import numpy as np
from collections import deque
from typing import List, Dict

class MetricsCollector:
    """
    Handles all metrics collection and calculation
    """
    
    def __init__(self, warmup_frames: int = 20, buffer_size: int = 100):
        self.warmup_frames = warmup_frames
        self.buffer_size = buffer_size
                
        # Metrics storage
        self.latencies: List[float] = []
        self.latency_buffer = deque(maxlen=buffer_size)
        
        # FPS tracking
        self.fps_history = deque(maxlen=30)
        self.fps_smooth = 0.0
        self.last_time = time.perf_counter()
        
        # Counters
        self.frame_count = 0
        self.start_time = None
    
    def add_frame_metrics(self, latency_ms: float) -> Dict:
        """
        Add metrics for one frame
        Returns current statistics
        """
        current_time = time.perf_counter()
        
        if self.start_time is None:
            self.start_time = current_time
        
        # Store latency after warmup
        if self.frame_count >= self.warmup_frames:
            self.latencies.append(latency_ms)
            self.latency_buffer.append(latency_ms)
        
        # Calculate FPS
        if self.last_time > 0:
            dt = current_time - self.last_time
            if dt > 0:
                instant_fps = 1.0 / dt
                self.fps_history.append(instant_fps)
                
                # Smooth FPS (matching original alpha=0.1)
                alpha = 0.1
                if self.fps_smooth > 0:
                    self.fps_smooth = (1 - alpha) * self.fps_smooth + alpha * instant_fps
                else:
                    self.fps_smooth = instant_fps
        
        self.last_time = current_time
        self.frame_count += 1
        
        return self.get_current_stats()
    
    def get_current_stats(self) -> Dict:
        """Get real-time statistics"""
        stats = {
            'frame_count': self.frame_count,
            'fps_smooth': self.fps_smooth,
            'is_warming_up': self.frame_count < self.warmup_frames
        }
        
        if self.latency_buffer and len(self.latency_buffer) > 10:
            stats.update({
                'mean_latency': float(np.mean(self.latency_buffer)),
                'p95_latency': float(np.percentile(self.latency_buffer, 95)),
                'p50_latency': float(np.percentile(self.latency_buffer, 50))
            })
        
        return stats
    
    def get_final_metrics(self) -> Dict:
        """
        Calculate final metrics
        """
        if not self.latencies:
            return {}
        
        latencies_np = np.array(self.latencies)
        
        # Original required metrics
        metrics = {
            'mean_latency_ms': float(np.mean(latencies_np)),
            'p95_latency_ms': float(np.percentile(latencies_np, 95)),
            'frames': len(self.latencies),
            'fps': len(self.latencies) / (np.sum(latencies_np) / 1000.0)
        }
        
        # Enhanced metrics
        metrics.update({
            'std_latency_ms': float(np.std(latencies_np)),
            'min_latency_ms': float(np.min(latencies_np)),
            'max_latency_ms': float(np.max(latencies_np)),
            'p50_latency_ms': float(np.percentile(latencies_np, 50)),
            'p90_latency_ms': float(np.percentile(latencies_np, 90)),
            'p99_latency_ms': float(np.percentile(latencies_np, 99)),
            'total_time_s': float(np.sum(latencies_np) / 1000.0),
        })
        
        return metrics