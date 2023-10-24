import cv2
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

camera = cv2.VideoCapture(0)
while camera.isOpened():
    success, frame = camera.read()
    if not success: break

    results = model(frame)
    print(results[0].boxesq)
    '''
    for i in results:
        for(x1,y1,x2,y2), cls, conf in zip(i.boxes.xyxy, i.boxes.cls, i.boxes.conf):
            x1,y1,x2,y2,cls=map(int,[x1,y1,x2,y2,cls])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,"%s %2.f" %(i.names[cls],conf),(x1,y1+20),0,0.7,(0,255,0),2)
    '''
    cv2.imshow("frame",frame)
    key_code=cv2.waitKey(1)
    if key_code in [27,ord('q')]:
        break
camera.release()
cv2.destroyAllWindows()
