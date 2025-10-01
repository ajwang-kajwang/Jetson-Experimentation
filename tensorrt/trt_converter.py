# trt_converter.py
"""
Converts PyTorch YOLO models to TensorRT engines using ONNX intermediate format
"""
import tensorrt as trt
from pathlib import Path
from typing import Optional, Tuple
from ultralytics import YOLO

# Import from baseline
import sys
sys.path.append(str(Path(__file__).parent.parent))
from baseline import config


class TensorRTConverter:
    """
    Convert PyTorch model to TensorRT    
    """
    
    def __init__(self, precision: str = "FP16"):
        self.precision = precision
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Use config paths
        self.model_path = config.MODEL_PT
        self.img_size = config.IMG_SIZE
        
        # Select paths based on precision 
        if precision == "FP16":
            self.onnx_path = config.ONNX_PATH_FP16
            self.engine_path = config.ENGINE_PATH_FP16
        else:
            self.onnx_path = config.ONNX_PATH
            self.engine_path = config.ENGINE_PATH
            
    def export_to_onnx(self) -> Path:
        """Step 1: Export PyTorch to ONNX"""
        print(f"Exporting {self.model_path} to ONNX...")
        model = YOLO(self.model_path)
        
        model.export(
            format='onnx',
            imgsz=self.img_size,
            simplify=True,
            half=(self.precision == "FP16")
        )
        
        print(f"ONNX saved to {self.onnx_path}")
        return self.onnx_path
    
    def build_engine(self) -> Optional[Path]:
        """ Build TensorRT engine from ONNX"""
        print(f"Building TensorRT engine...")
        
        builder = trt.Builder(self.logger)
        config = builder.create_builder_config()
        config.max_workspace_size = 4 * (1 << 30)  # 4GB
        
        # Set precision flag
        if self.precision == "FP16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        
        # Parse ONNX
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)
        
        with open(self.onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                return None
        
        # Build and save engine
        engine = builder.build_engine(network, config)
        if engine:
            with open(self.engine_path, 'wb') as f:
                f.write(engine.serialize())
            print(f"Engine saved to {self.engine_path}")
            return self.engine_path
        return None
    
    def convert(self) -> Tuple[bool, Path]:
        """Complete conversion pipeline"""
        try:
            self.export_to_onnx()
            result = self.build_engine()
            return (True, result) if result else (False, None)
        except Exception as e:
            print(f"Conversion failed: {e}")
            return False, None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--precision', default='FP16', choices=['FP32', 'FP16'])
    args = parser.parse_args()
    
    converter = TensorRTConverter(args.precision)
    success, engine_path = converter.convert()
    
    if success:
        print(f"✓ Success: {engine_path}")
    else:
        print("✗ Failed")
        exit(1)