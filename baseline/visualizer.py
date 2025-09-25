# baseline/visualizer.py
import cv2
import numpy as np
from typing import Dict

class Visualizer:
    """Handles visualization and display"""
    
    def __init__(self, window_name: str = "PyTorch YOLOv8 (Baseline)"):
        self.window_name = window_name
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.colors = {
            'white': (255, 255, 255),
            'green': (0, 255, 0),
            'yellow': (255, 255, 0),
            'red': (0, 0, 255)
        }
    
    def draw_metrics(self, frame: np.ndarray, stats: Dict[str, float], 
                    warming_up: bool = False) -> np.ndarray:
        """Draw metrics overlay on frame"""
        annotated = frame.copy()
        
        # Match original simple display
        if 'current_latency' in stats and 'fps_smooth' in stats:
            text = f"Latency: {stats['current_latency']:.1f} ms  FPS: {stats['fps_smooth']:.1f}"
            cv2.putText(annotated, text, (10, 30), 
                       self.font, 0.9, self.colors['green'], 2, cv2.LINE_AA)
        
        # Add enhanced metrics below
        y_offset = 60
        if 'mean_latency' in stats:
            cv2.putText(annotated, f"Mean: {stats['mean_latency']:.1f} ms", 
                       (10, y_offset), self.font, 0.6, self.colors['yellow'], 2)
            y_offset += 25
            
            if 'p95_latency' in stats:
                cv2.putText(annotated, f"P95: {stats['p95_latency']:.1f} ms", 
                           (10, y_offset), self.font, 0.6, self.colors['yellow'], 2)
        
        if warming_up:
            h, w = annotated.shape[:2]
            cv2.putText(annotated, "WARMING UP...", 
                       (w//2 - 100, 30), self.font, 1.0, self.colors['red'], 3)
        
        return annotated
    
    def show(self, frame: np.ndarray) -> bool:
        """Display frame and return True if should continue"""
        cv2.imshow(self.window_name, frame)
        return cv2.waitKey(1) & 0xFF != 27  # ESC to exit