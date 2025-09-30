 # LAB_DIR/task1_infer_trt.py
import time
import numpy as np
import cv2
import argparse
import json
import yaml
from pathlib import Path
from ultralytics import YOLO
from config import ENGINE_PATH, IMG_SIZE, CAMERA_INDEX, DATA_YAML

# Fix deprecated np.bool alias
if not hasattr(np, "bool"):
    np.bool = np.bool_

def xyxy_iou(box1, box2):
    """Compute IoU between two sets of boxes in xyxy format."""
    if len(box1) == 0 or len(box2) == 0:
        return np.zeros((len(box1), len(box2)), dtype=np.float32)

    x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
    y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
    x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
    y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])

    inter_area = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = box1_area[:, None] + box2_area[None, :] - inter_area
    iou = inter_area / np.clip(union_area, 1e-6, None)
    return iou

def load_yolo_dataset(yaml_file):
    """Load YOLO dataset from YAML file, supporting relative label folder."""
    with open(yaml_file, 'r') as f:
        data_cfg = yaml.safe_load(f)

    dataset_root = Path(data_cfg['path']).resolve()
    val_file = Path(data_cfg['val']).resolve()
    if not val_file.exists():
        val_file = dataset_root / data_cfg['val']
    if not val_file.exists():
        raise FileNotFoundError(f"Validation file not found: {val_file}")

    dataset = []
    with open(val_file, 'r') as f:
        lines = f.read().splitlines()

    for line in lines:
        img_path = (dataset_root / line).resolve()
        if not img_path.exists():
            print(f"Warning: image not found: {img_path}")
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Correct relative label path
        label_path = (img_path.parent.parent.parent / "labels/val2017" / img_path.name).with_suffix('.txt')
        boxes = []
        if label_path.exists():
            with open(label_path, 'r') as lf:
                for lbl in lf:
                    cls, x_center, y_center, bw, bh = map(float, lbl.strip().split())
                    x1 = (x_center - bw/2) * w
                    y1 = (y_center - bh/2) * h
                    x2 = (x_center + bw/2) * w
                    y2 = (y_center + bh/2) * h
                    boxes.append([x1, y1, x2, y2])
        else:
            print(f"Warning: label file not found: {label_path}")

        dataset.append((str(img_path), np.array(boxes)))

    print(f"Found {len(dataset)} validation images.")
    return dataset

def infer_live(camera_index: int, metrics_out: str = 'trt_live_metrics.json'):
    assert ENGINE_PATH.exists(), f'TensorRT engine not found: {ENGINE_PATH}'
    model = YOLO(str(ENGINE_PATH))

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f'Camera {camera_index} not available')

    latencies = []
    t_last, fps_smooth, alpha = time.perf_counter(), 0.0, 0.1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        results = model.predict(source=frame, imgsz=IMG_SIZE, verbose=False, device=0)
        t1 = time.perf_counter()

        lat_ms = (t1 - t0) * 1000.0
        latencies.append(lat_ms)

        annotated = results[0].plot()
        inst_fps = 1.0 / max(t1 - t_last, 1e-6)
        fps_smooth = inst_fps if fps_smooth == 0 else (1 - alpha) * fps_smooth + alpha * inst_fps
        t_last = t1

        cv2.putText(
            annotated,
            f'Latency: {lat_ms:.1f} ms  FPS: {fps_smooth:.1f}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        cv2.imshow('TensorRT YOLOv8x', annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if latencies:
        metrics = {
            'mean_latency_ms': float(np.mean(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'frames': len(latencies),
            'fps': len(latencies) / (np.sum(latencies) / 1000.0)
        }
    else:
        metrics = {
            'mean_latency_ms': float('nan'),
            'p95_latency_ms': float('nan'),
            'frames': 0,
            'fps': float('nan')
        }

    print(metrics)
    with open(metrics_out, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved live metrics to {metrics_out}")

def infer_coco(metrics_out: str = 'trt_coco_metrics.json'):
    assert ENGINE_PATH.exists(), f'TensorRT engine not found: {ENGINE_PATH}'
    model = YOLO(str(ENGINE_PATH))

    dataset = load_yolo_dataset(DATA_YAML)

    if len(dataset) == 0:
        print("Warning: No validation images found!")
        metrics = {
            'precision': float('nan'),
            'recall': float('nan'),
            'f1': float('nan'),
            'mean_latency_ms': float('nan'),
            'p95_latency_ms': float('nan'),
            'frames': 0,
            'fps': float('nan')
        }
        with open(metrics_out, 'w') as f:
            json.dump(metrics, f, indent=2)
        return

    latencies = []
    all_precisions, all_recalls = [], []

    for img_path, gt_boxes in dataset:
        t0 = time.perf_counter()
        results = model.predict(source=img_path, imgsz=IMG_SIZE, verbose=False, device=0)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        pred_boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else np.zeros((0, 4))

        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            ious = xyxy_iou(pred_boxes, gt_boxes)
            tp = (ious >= 0.5).sum()
            fp = len(pred_boxes) - tp
            fn = len(gt_boxes) - tp
            precision = tp / max(tp + fp, 1e-6)
            recall = tp / max(tp + fn, 1e-6)
        else:
            precision, recall = 0.0, 0.0

        all_precisions.append(precision)
        all_recalls.append(recall)

    mean_precision = float(np.mean(all_precisions))
    mean_recall = float(np.mean(all_recalls))
    f1 = 2 * mean_precision * mean_recall / max(mean_precision + mean_recall, 1e-12)
    mean_latency = float(np.mean(latencies))
    fps = len(latencies) / (np.sum(latencies) / 1000.0)

    metrics = {
        'precision': mean_precision,
        'recall': mean_recall,
        'f1': f1,
        'mean_latency_ms': mean_latency,
        'p95_latency_ms': float(np.percentile(latencies, 95)),
        'frames': len(latencies),
        'fps': fps
    }

    print(metrics)
    with open(metrics_out, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved dataset metrics to {metrics_out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TensorRT YOLOv8 inference")
    parser.add_argument('--mode', choices=['live', 'coco'], default='coco')
    parser.add_argument('--camera', type=int, default=CAMERA_INDEX)
    parser.add_argument('--metrics_out', type=str, default=None)
    args = parser.parse_args()

    out_file = args.metrics_out or ('trt_live_metrics.json' if args.mode == 'live' else 'trt_coco_metrics.json')
    if args.mode == 'live':
        infer_live(args.camera, metrics_out=out_file)
    else:
        infer_coco(metrics_out=out_file)