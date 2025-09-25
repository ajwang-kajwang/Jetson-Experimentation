# baseline/metrics.py
import time
import numpy as np
from collections import deque
from typing import List, Dict

class MetricsCollector:
    """Collects and calculates performance metrics"""
    
    def __init__(self, warmup_frames: int = 20, buffer_size: int = 100):
        self.warmup_frames = warmup_frames
        self.buffer_size = buffer_size
        self.latencies: List[float] = []
        self.latency_buffer = deque(maxlen=buffer_size)
        self.fps_history = deque(maxlen=30)
        self.fps_smooth = 0.0
        self.frame_count = 0
        self.last_time = time.perf_counter()
    
    def add_measurement(self, latency_ms: float, timestamp: float) -> None:
        """Add a new latency measurement"""
        if self.frame_count >= self.warmup_frames:
            self.latencies.append(latency_ms)
            self.latency_buffer.append(latency_ms)
        
        dt = timestamp - self.last_time
        if dt > 0:
            instant_fps = 1.0 / dt
            self.fps_history.append(instant_fps)
            alpha = 0.1
            if self.fps_smooth > 0:
                self.fps_smooth = (1 - alpha) * self.fps_smooth + alpha * instant_fps
            else:
                self.fps_smooth = instant_fps
        
        self.last_time = timestamp
        self.frame_count += 1
    
    def is_warming_up(self) -> bool:
        """Check if still in warmup period"""
        return self.frame_count < self.warmup_frames
    
    def get_current_stats(self) -> Dict[str, float]:
        """Get current rolling statistics"""
        stats = {
            'frame_count': self.frame_count,
            'fps_smooth': self.fps_smooth,
            'fps_avg': np.mean(self.fps_history) if self.fps_history else 0
        }
        
        if len(self.latency_buffer) > 10:
            stats.update({
                'mean_latency': np.mean(self.latency_buffer),
                'p95_latency': np.percentile(self.latency_buffer, 95),
                'p50_latency': np.percentile(self.latency_buffer, 50)
            })
        
        return stats
    
    def calculate_final_metrics(self) -> Dict[str, float]:
        """Calculate final metrics - matches original output format"""
        if not self.latencies:
            return {}
        
        latencies_np = np.array(self.latencies)
        
        # Match original metrics structure
        metrics = {
            'mean_latency_ms': float(np.mean(latencies_np)),
            'p95_latency_ms': float(np.percentile(latencies_np, 95)),
            'frames': len(self.latencies),
            'fps': len(self.latencies) / (np.sum(latencies_np) / 1000.0)
        }
        
        # Add enhanced metrics
        metrics.update({
            'std_latency_ms': float(np.std(latencies_np)),
            'min_latency_ms': float(np.min(latencies_np)),
            'max_latency_ms': float(np.max(latencies_np)),
            'p50_latency_ms': float(np.percentile(latencies_np, 50)),
            'p90_latency_ms': float(np.percentile(latencies_np, 90)),
            'p99_latency_ms': float(np.percentile(latencies_np, 99)),
        })
        
        if len(self.latencies) > 1:
            diffs = np.diff(latencies_np)
            metrics['jitter_ms'] = float(np.std(diffs))
        
        return metrics