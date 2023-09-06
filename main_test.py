import cv2
import pandas as pd
import random
from ultralytics import YOLO
# from tracker import*
from deepsort import Tracker

model=YOLO('yolov8n-face.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('People 3.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0
up_count = 0

people_list = {}
counted = set()

tracker=Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

cy1=322
cy2=368
offset=6

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
    
    results = model.predict(frame)
    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            class_id = int(class_id)
            detections.append([x1, y1, x2, y2, score])
        tracker.update(frame, detections)
        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            track_id = track.track_id

            if y1 > cy1:
                people_list[track_id] = y1
            
            cv2.putText(frame, (str(track_id)),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,255),2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (colors[track_id % len(colors)]), 3)

    cv2.putText(frame, ("Count :"+str(len(people_list))),(500,cy1),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,255),2)
    
    cv2.line(frame,(0,cy1),(1280,cy1),(255,255,255),1)
    cv2.putText(frame, ("Line 1"),(280,cy1),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)


    cv2.line(frame,(0,cy2),(1280,cy2),(0,0,255),4)
    cv2.putText(frame, ("Line 2"),(100,cy2-20),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,0),2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

