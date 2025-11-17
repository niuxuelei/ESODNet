import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/ESODNet/runs/train/exp_yolov8n-FDPN-LSCD-DOTA-v1.0/weights/best.pt')
    model.val(data='/ESODNet/ultralytics/cfg/datasets/DOTA-v1.0.yaml',
              split='val',
              imgsz=640,
              batch=8,
              # rect=False,
              save_json=True, 
              project='runs/val',
              name='exp_yolov8n-FDPN-LSCD-DOTA-v1.0',
              )



