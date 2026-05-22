from ultralytics import YOLO
import cv2
model = YOLO("/home/jasstaran/Documents/ObjDet/venv/objDetection/yolo-weights/yolo11x.pt")
results = model("/home/jasstaran/Documents/ObjDet/venv/objDetection/img/school.jpg")
results[0].show()
