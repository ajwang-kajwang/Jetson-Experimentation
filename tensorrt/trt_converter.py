"""
TensorRT converter for YOLO models
Converts .pt or .onnx to .engine
"""
import tensorrt as trt
from pathlib import Path


def convert_onnx_to_engine(onnx_path, engine_path, fp16=True):
    """
    Convert ONNX model to TensorRT engine
    Args:
        onnx_path: Path to ONNX model
        engine_path: Output engine path
        fp16: Use FP16 precision
    """
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    
    # Set max workspace size (4GB)
    config.max_workspace_size = 4 * (1 << 30)
    
    # Enable FP16 if requested
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("Using FP16 precision")
    
    # Parse ONNX
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    
    print(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    # Build engine
    print("Building TensorRT engine... This may take a few minutes")
    engine = builder.build_engine(network, config)
    
    if engine is None:
        print("Failed to build engine")
        return False
    
    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"Engine saved to: {engine_path}")
    print(f"Engine size: {Path(engine_path).stat().st_size / 1e6:.1f} MB")
    return True


def convert_pt_to_engine(pt_path, engine_path, fp16=True):
    """
    Convert PyTorch model to TensorRT engine via ONNX
    Args:
        pt_path: Path to .pt model
        engine_path: Output engine path
        fp16: Use FP16 precision
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed. Install with: pip install ultralytics")
        return False
    
    # Export to ONNX first
    print(f"Loading PyTorch model: {pt_path}")
    model = YOLO(pt_path)
    
    onnx_path = Path(pt_path).with_suffix('.onnx')
    print(f"Exporting to ONNX: {onnx_path}")
    
    model.export(format='onnx', imgsz=640, simplify=True, half=fp16)
    
    # Convert ONNX to TensorRT
    return convert_onnx_to_engine(onnx_path, engine_path, fp16)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert model to TensorRT')
    parser.add_argument('model', type=str, help='Input model (.pt or .onnx)')
    parser.add_argument('--output', type=str, help='Output engine path')
    parser.add_argument('--fp32', action='store_true', help='Use FP32 (default: FP16)')
    
    args = parser.parse_args()
    
    # Check input exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        exit(1)
    
    # Determine output path
    if args.output:
        engine_path = Path(args.output)
    else:
        precision = 'fp32' if args.fp32 else 'fp16'
        engine_path = model_path.with_name(f"{model_path.stem}_{precision}.engine")
    
    # Convert based on file type
    use_fp16 = not args.fp32
    
    if model_path.suffix == '.onnx':
        success = convert_onnx_to_engine(model_path, engine_path, use_fp16)
    elif model_path.suffix == '.pt':
        success = convert_pt_to_engine(model_path, engine_path, use_fp16)
    else:
        print(f"Unsupported model format: {model_path.suffix}")
        print("Supported: .pt (PyTorch) or .onnx")
        exit(1)
    
    exit(0 if success else 1)