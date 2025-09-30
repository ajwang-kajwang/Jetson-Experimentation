# LAB_DIR/task1_export_onnx.py
from ultralytics import YOLO
from pathlib import Path
from config import MODEL_PT, IMG_SIZE, DYNAMIC, HALF, ONNX_PATH

def main():
    model = YOLO(MODEL_PT)
    exported = model.export(format='onnx', imgsz=IMG_SIZE, dynamic=DYNAMIC, half=HALF)
    src = Path(str(exported))
    if src.exists() and src.resolve() != ONNX_PATH.resolve():
        src.rename(ONNX_PATH)
    print(f'ONNX saved at: {ONNX_PATH}')

if __name__ == '__main__':
    main()

