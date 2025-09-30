 
# LAB_DIR/task1_build_trt_engine.py
from ultralytics.utils.export import export_engine
from config import ONNX_PATH, ENGINE_PATH, BATCH, IMG_SIZE, DYNAMIC

def main():
    assert ONNX_PATH.exists(), 'ONNX file not found'
    export_engine(
        onnx_file=str(ONNX_PATH),
        engine_file=str(ENGINE_PATH),
        workspace=4,
        half=False,
        int8=False,
        dynamic=DYNAMIC,
        shape=(BATCH, 3, IMG_SIZE, IMG_SIZE),
        verbose=True
    )
    print(f'Engine saved at: {ENGINE_PATH}')

if __name__ == '__main__':
    main()
