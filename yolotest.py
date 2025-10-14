from ultralytics import YOLO
import cv2
from collections import Counter
model = YOLO("v8_detect_guns.pt")
cap = cv2.VideoCapture(0)
ok, frame = cap.read()
cap.release()
if not ok:
    print("❌ Failed to capture image from webcam.")
else:
    results = model.predict(frame, conf=0.4)[0]
    names = model.model.names if hasattr(model.model, "names") else model.names
    objs = [names[int(b.cls)] for b in results.boxes]
    print("👁 Detected objects:", (objs))