# trt_inference.py
"""
Task 1: TensorRT Inference Engine
Runs inference using TensorRT engine, inherits from baseline
"""
import time
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
from typing import Dict, List

# Import baseline
import sys
sys.path.append(str(Path(__file__).parent.parent))
from baseline import config
from baseline.baseline_infer_pytorch import BaselineInferenceEngine

class TRTRuntime:
    """Handles TensorRT-specific operations"""
    
    def __init__(self, engine_path: Path):
        # Load engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
    
    def infer(self, image: np.ndarray) -> List[np.ndarray]:
        """Run inference on image"""
        # Preprocess: resize, RGB, normalize, CHW, batch
        img = cv2.resize(image, (config.IMG_SIZE, config.IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
        
        # Copy to GPU
        np.copyto(self.inputs[0]['host'], img.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], 
                               self.inputs[0]['host'], self.stream)
        
        # Execute
        self.context.execute_async_v2(bindings=self.bindings, 
                                      stream_handle=self.stream.handle)
        
        # Copy results
        results = []
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)
            results.append(output['host'].reshape(output['shape']))
        
        self.stream.synchronize()
        return results


class TensorRTInferenceEngine(BaselineInferenceEngine):
    """
    Task 1: Inherit from baseline, override only what's needed for TensorRT
    """
    
    def __init__(self, precision: str = "FP16"):
        super().__init__()  # Use parent's initialization
        
        # Select engine path based on precision
        self.engine_path = (config.ENGINE_PATH_FP16 if precision == "FP16" 
                           else config.ENGINE_PATH)
        self.precision = precision
        self.trt_runtime = None
        
        # Update window name
        self.visualizer.window_name = "TensorRT YOLOv8"
    
    def load_model(self) -> None:
        """Override: Load TensorRT engine instead of PyTorch model"""
        print(f"Loading TensorRT engine: {self.engine_path}")
        
        if not self.engine_path.exists():
            raise FileNotFoundError(f"Engine not found. Run trt_converter.py first")
            
        self.trt_runtime = TRTRuntime(self.engine_path)
        print("TensorRT engine loaded")
    
    def run_live_inference(self, camera_index: int = None, 
                          display: bool = True) -> Dict[str, float]:
        """Use TensorRT for inference"""
        if camera_index is None:
            camera_index = self.config.CAMERA_INDEX
        
        self.print_system_info()
        self.load_model()
        
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.IMG_SIZE)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.IMG_SIZE)
        
        print(f"\nStarting TensorRT inference... Press ESC to stop")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # ONLY DIFFERENCE: Use TensorRT instead of PyTorch
                t_start = time.perf_counter()
                outputs = self.trt_runtime.infer(frame)
                t_end = time.perf_counter()
                
                latency_ms = (t_end - t_start) * 1000.0
                
                # Rest uses parent's metrics collection
                self.metrics_collector.add_measurement(latency_ms, t_end)
                stats = self.metrics_collector.get_current_stats()
                stats['current_latency'] = latency_ms
                
                if display:
                    # Simple visualization - could enhance with detections
                    annotated = self.visualizer.draw_metrics(
                        frame, stats,
                        warming_up=self.metrics_collector.is_warming_up()
                    )
                    
                    if not self.visualizer.show(annotated):
                        break
                
                if self.metrics_collector.frame_count % 100 == 0:
                    self._print_periodic_stats(stats)
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            metrics = self.metrics_collector.calculate_final_metrics()
            metrics['engine'] = 'TensorRT'
            metrics['precision'] = self.precision
            return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--precision', default='FP16', choices=['FP32', 'FP16'])
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--output', default='trt_metrics.json')
    args = parser.parse_args()
    
    engine = TensorRTInferenceEngine(args.precision)
    metrics = engine.run_live_inference(args.camera)
    engine.save_metrics(metrics, args.output)