import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*

model=YOLO('yolov8s.pt')



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('People.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0
up_count = 0

people_list = {}
counted = set()

tracker=Tracker()

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
   

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
             
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'person' in c:
            list.append([x1,y1,x2,y2])
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        if cy1<(cy+offset) and cy1>(cy-offset):
            people_list[id] = cy
        if id in people_list:
            if cy2<(cy+offset) and cy2>(cy-offset):
                up_count+=1
                counted.add(id)
                # print(counted)
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
           
    # cv2.putText(frame, ("Count :"+str(len(people_list))),(500,cy1),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,255),2)
    cv2.putText(frame, ("Count : "+str(len(counted))),(500,cy1),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,255),2)



    # cv2.line(frame,(0,cy1),(1280,cy1),(255,255,255),1)
    # cv2.putText(frame, ("Line 1"),(280,cy1),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)


    cv2.line(frame,(0,cy2),(1280,cy2),(0,0,255),4)
    cv2.putText(frame, ("Line 2"),(100,cy2-20),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,0),2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

