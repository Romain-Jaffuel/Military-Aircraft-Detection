# Military Aircraft Detection with YOLOv8

This project trains a YOLOv8 model to detect and classify military aircraft across 85 different classes.

The dataset consists of images and associated csv files found on Kaggle at https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset, each containing bounding box annotations. These annotations were converted into YOLOv8 format and organized into appropriate folders: `images/train` and `labels/train`.

Model training was performed using YOLOv8s with the following configuration:
- 30 epochs
- batch size of 16
- image resolution of 640x640
- image caching disabled due to RAM limitations

The best model was saved to `YOLOv8s_trained/weights/best.pt`.

Performance metrics on the validation set:
Precision: 0.944  
Recall: 0.878  
mAP@0.5: 0.945  
mAP@0.5:0.95: 0.859

To run inference on a new image:

```python
from ultralytics import YOLO

model = YOLO("YOLOv8s_trained/weights/best.pt")
results = model.predict(
    source="path/to/your/image",
    conf=0.25,
    save=True,
    show_labels=True,
    show_conf=True
)
