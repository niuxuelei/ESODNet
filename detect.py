import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/ps/ultralytics-202404066/ultralytics-main/runs/train/exp_yolov8n-FDPN-LSCD1_nwpu2/weights/best.pt') # select your model.pt path
    model.predict(source='/home/ps/ultralytics-202404066/ultralytics-main/dataset/NWPUVHR/images/test',
                  imgsz=640,
                  project='runs/detect',
                  name='exp_yolov8n-FDPN-LSCD1_nwpu111',
                  save=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )