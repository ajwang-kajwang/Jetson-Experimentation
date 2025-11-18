"""
TensoRT Inference
"""
import time
import cv2
import numpy as np
import tensorrt as trt
from pathlib import Path


class TensorRTInference:
    """TensorRT inference"""
    
    def __init__(self, engine_path):
        """Load TensorRT engine and allocate buffers"""
        # Load engine
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.buffers = []
        for i in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(i)
            size = int(np.prod(shape))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            buffer = np.empty(size, dtype)
            self.buffers.append(buffer)
        
        print(f"Engine loaded: {self.engine.num_bindings} bindings")
    
    def infer(self, input_array):
        """
        Run inference
        Args:
            input_array: Preprocessed numpy array
        Returns:
            Output arrays
        """
        # Copy input to buffer
        np.copyto(self.buffers[0], input_array.ravel())
        
        # Run inference
        bindings = [buf.ctypes.data for buf in self.buffers]
        self.context.execute_v2(bindings)
        
        # Return outputs (usually just one for YOLO)
        return self.buffers[1].copy()


def preprocess_image(image, size=640):
    """
    Preprocess image for YOLO inference
    Args:
        image: BGR image from OpenCV
        size: Target size
    Returns:
        Preprocessed array (1,3,640,640)
    """
    # Resize
    img = cv2.resize(image, (size, size))
    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize to [0,1]
    img = img.astype(np.float32) / 255.0
    # HWC to CHW
    img = np.transpose(img, (2, 0, 1))
    # Add batch dimension
    img = np.expand_dims(img, 0)
    return img


def run_camera_inference(engine_path, camera_index=0):
    """
    Run inference on camera feed
    Args:
        engine_path: Path to .engine file
        camera_index: Camera device index
    """
    # Initialize TensorRT
    trt_engine = TensorRTInference(engine_path)
    
    # Open camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Failed to open camera {camera_index}")
        return
    
    print("Starting inference... Press ESC to stop")
    
    frame_count = 0
    total_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess
            input_tensor = preprocess_image(frame)
            # Inference
            t_start = time.perf_counter()
            output = trt_engine.infer(input_tensor)
            t_end = time.perf_counter()
            
            # Calculate metrics
            latency_ms = (t_end - t_start) * 1000
            total_time += latency_ms
            frame_count += 1
            fps = 1000 / latency_ms if latency_ms > 0 else 0
            
            # Display metrics on frame
            text = f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms"
            cv2.putText(frame, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow("TensorRT Inference", frame)
            
            # Check for ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break
            
            # Print periodic updates
            if frame_count % 30 == 0:
                avg_latency = total_time / frame_count
                print(f"Frame {frame_count}: Avg latency: {avg_latency:.1f}ms")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        if frame_count > 0:
            avg_latency = total_time / frame_count
            avg_fps = 1000 / avg_latency
            print(f"\nResults: {frame_count} frames")
            print(f"Average latency: {avg_latency:.1f}ms")
            print(f"Average FPS: {avg_fps:.1f}")


def run_image_inference(engine_path, image_path):
    """
    Run inference on single image
    Args:
        engine_path: Path to .engine file
        image_path: Path to input image
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Initialize TensorRT
    trt_engine = TensorRTMinimal(engine_path)
    
    # Preprocess
    input_tensor = preprocess_image(image)
    
    # Inference
    t_start = time.perf_counter()
    output = trt_engine.infer(input_tensor)
    t_end = time.perf_counter()
    
    latency_ms = (t_end - t_start) * 1000
    
    print(f"Inference completed:")
    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Latency: {latency_ms:.2f}ms")
    
    return output


def benchmark(engine_path, num_iterations=100):
    """
    Benchmark TensorRT engine
    Args:
        engine_path: Path to .engine file
        num_iterations: Number of iterations
    """
    # Initialize
    trt_engine = TensorRTMinimal(engine_path)
    
    # Create random input
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        trt_engine.infer(dummy_input)
    
    # Benchmark
    latencies = []
    for i in range(num_iterations):
        t_start = time.perf_counter()
        trt_engine.infer(dummy_input)
        t_end = time.perf_counter()
        latencies.append((t_end - t_start) * 1000)
        
        if (i + 1) % 20 == 0:
            print(f"Iteration {i+1}/{num_iterations}")
    
    # Calculate stats
    latencies = np.array(latencies)
    print(f"\nBenchmark Results ({num_iterations} iterations):")
    print(f"  Mean: {np.mean(latencies):.2f}ms")
    print(f"  Std: {np.std(latencies):.2f}ms")
    print(f"  Min: {np.min(latencies):.2f}ms")
    print(f"  Max: {np.max(latencies):.2f}ms")
    print(f"  P50: {np.percentile(latencies, 50):.2f}ms")
    print(f"  P95: {np.percentile(latencies, 95):.2f}ms")
    print(f"  P99: {np.percentile(latencies, 99):.2f}ms")
    print(f"  FPS: {1000/np.mean(latencies):.1f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TensorRT Inference')
    parser.add_argument('--engine', type=str, default='yolov8x_fp16.engine') # Path to TensorRT engine
    parser.add_argument('--camera', type=int) # Camera index for live inference
    parser.add_argument('--image', type=str)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--iterations', type=int, default=100)    
    args = parser.parse_args()
    
    # Check engine exists
    if not Path(args.engine).exists():
        print(f"Engine not found: {args.engine}")
        print("Run trt_converter.py first to create engine")
        exit(1)
    
    # Mode Selection
    if args.benchmark:
        benchmark(args.engine, args.iterations)
    elif args.image:
        run_image_inference(args.engine, args.image)
    else:
        # Default to camera
        camera_idx = args.camera if args.camera is not None else 0
        run_live_inference(args.engine, camera_idx)