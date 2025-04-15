import os
from ultralytics import YOLO
import cv2

class RKNNDetection:
    def __init__(self, model_path = r"./best_rknn_model", input_path = "./demo/demo.jpg",output_path = "./results", confidence_thres = 0.5, iou_thres = 0.45):
        self.model = YOLO(model_path, task = 'detect')
        self.input_path = input_path
        self.conf = confidence_thres
        self.iou = iou_thres
        

    def detect(self):
        if self.input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
            cap = cv2. VideoCapture(self.input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('./results/rknn_run/rknn_video.mp4',fourcc, fps, (width,height))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                new_frame = self.model.predict(frame, imgsz=640, conf=self.conf, iou=self.iou)[0].plot()
                out.write(new_frame)
            cap.release()
            out.release()
        else:
            self.model(self.input_path, imgsz=640, conf = self.conf, iou =self.iou, save = True, project = 'results', name = 'rknn_run')
        
    def detect_frame(self,frame):
        return self.model(frame, imgsz=640, conf=self.conf, iou=self.iou, project = 'results', name = 'rknn')[0].plot()

if __name__ == '__main__':
    RKNNDetection().detect()