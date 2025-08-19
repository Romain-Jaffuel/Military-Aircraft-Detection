# Military Aircraft Detection — YOLOv8
This project explores how to improve recognition across dozens of aircraft classes by enriching training data beyond Kaggle, isolating the effect of adding auto-labeled video frames to highlight the role of dataset diversity.

## Context & Pipeline

1) **V1 — Model trained on images (Kaggle)**  
   - Source: `a2015003713/militaryaircraftdetectiondataset` (Kaggle). Imported via `kagglehub` in the notebook `Military_Aircraft_Detection.ipynb`.  
   - Conversion to YOLO: the CSVs containing `xmin,ymin,xmax,ymax,class` are transformed into `.txt` (YOLO format) and paired with the corresponding `.jpg`. The notebook script (section “dataset → yolo_dataset”) creates `yolo_dataset/images/train` and `yolo_dataset/labels/train`.  
   - Training (Ultralytics): `yolov8s.pt`, `epochs=30`, `imgsz=640`, `batch=16`, `close_mosaic=10`, `mixup=0.0`, `copy_paste=0.0`.  
     Directory: `YOLOv8s_trained_on_images/` (weights: `weights/best.pt`, logs: `args.yaml`, `results.csv`).

2) **Auto-labeling of videos (Haci Productions) with V1**  
   - Videos per aircraft type downloaded from the YouTube channel `@HaciProductions` (`yt_dlp`).  
   - For each aircraft folder (e.g. `aircraft_video/J-20/*.mp4`): extract frames, run inference with V1 (`conf=0.25`, `imgsz=640`), then save YOLO labels in folders `*_frames_labeled/` (e.g. `J-20_frames_labeled/`).  
   - Aggregation: a routine copies/renames only image+txt pairs into `yolo_dataset/images/train` and `yolo_dataset/labels/train`, boosting both volume and diversity per class.

3) **V2 — Model trained on images + auto-labeled videos**
   - Same training setup (Ultralytics) and hyperparameters as V1 (`epochs=30`, `imgsz=640`, `batch=16`, `close_mosaic=10`, `mixup=0.0`).  
   - Directory: `YOLOv8S_trained_on_images+videos/` (weights: `weights/best.pt`, logs: `args.yaml`, `results.csv`).

> Classes: about **85 categories** (from `data.yaml` and Kaggle class names after normalization).

## Evaluation on common holdout
To compare fairly, we **sample K images per class** from the current dataset (`yolo_dataset`) and evaluate **V1** and **V2** on the **exact same list of images** (common holdout). The script outputs `comparison_common_holdout.csv` with metrics.

- Ultralytics eval settings: `imgsz=640`, `conf=0.001`, `iou=0.6`.  
- Both models expose `nc≈85` (consistent with `data.yaml`).

### Results
| Model | F1 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Inference (ms/img) |
|---|---:|---:|---:|---:|---:|---:|
| **V2 — Images+Videos** (`YOLOv8S_trained_on_images+videos/weights/best.pt`) | **0.7864** | 0.8827 | 0.7090 | 0.7354 | 0.6699 | 0.7032 |
| **V1 — Images (Kaggle)** (`YOLOv8s_trained_on_images/weights/best.pt`) | 0.7267 | 0.8429 | 0.6387 | 0.6735 | 0.6072 | 0.6789 |

**Delta (V2 − V1)**:  
F1 **+0.0597**, Precision **+0.0397**, Recall **+0.0704**, mAP50 **+0.0619**, mAP50-95 **+0.0627**, speed +0.0061 ms/img.

**Interpretation (personal opinion)**: Adding video frames increases variability (angles, backgrounds, weather, resolution), which **improves recall in particular** and thus also boosts F1 and mAP. The runtime overhead is negligible.
The comparison CSV is versioned: `comparison_common_holdout.csv`.

## Quick test on a single image

You can run inference either from **Python** or from the **CLI**. Below are copy-pasteable examples for both **V2 (images+videos)** and **V1 (images only)** on Windows paths with spaces.

### Python (recommended)

```python
from ultralytics import YOLO

# --- Pick your model ---
# V2 (images + auto-labeled video frames)
model_path = r"YOLOv8S_trained_on_images+videos/weights/best.pt"
# V1 (images/Kaggle only)
# model_path = r"YOLOv8s_trained_on_images/weights/best.pt"

# --- Single image to test ---
image_path = r"test_data/demo.jpg"

# Load model
model = YOLO(model_path)

# Inference
results = model.predict(
    source=image_path,
    imgsz=640,           # keep 640 unless objects are tiny
    save=True, 
    project=r"predict",
    name="v2_single_image",
    exist_ok=True
)

# Print detections (class name, conf, box)
names = model.names
for r in results:
    for b in (r.boxes or []):
        cls = int(b.cls.item())
        conf = float(b.conf.item())
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        print(f"{names.get(cls, cls)}  conf={conf:.3f}  xyxy=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

print("Annotated image saved under the 'predict/v2_single_image' folder.")


## Closing Notes

This project provides a practical example of how combining open datasets with video mining can improve real-world object detection. Contributions are welcome