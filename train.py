import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11s-SDC.yaml')
    model.train(data='C:/Users/Zenobia/Desktop/v11-SDC/YOLOv11-SDC/ultralytics/cfg/datasets/mine_data.yaml',
                cache=False,
                imgsz=640,
                epochs=600,
                batch=32,
                close_mosaic=0,
                workers=4,
                # device='0',
                optimizer='SGD',
                patience=0,
                project='runs/SDC',
                name='exp',
    )