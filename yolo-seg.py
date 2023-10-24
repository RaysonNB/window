import cv2
from ultralytics import YOLO

image=cv2.imread("C:/Users/rayso_sq9ff\Downloads/a1108037060_65.png")
image = cv2.resize(image, (640, 480))

model = YOLO("yolov8n-seg.pt")

camera = cv2.VideoCapture(0)
while camera.isOpened():
    success, frame = camera.read()
    if not success: break

    results = model(frame)
    for r in results:
        if zip(r.boxes.cls, r.masks) is None: continue
        for cls, mask in zip(r.boxes.cls, r.masks):
            if cls != 0: continue
            mask = mask.data.cpu().numpy()[0]
            img = image.copy()
            for i in range(3):
                frame[:, :, i] = frame[:, :, i] * mask
                img[:, :, i] = img[:, :, i] * (1 - mask)
            frame = frame + img

    cv2.imshow("frame", frame)
    key_code = cv2.waitKey(1)
    if key_code in [27, ord('q')]:
        break
camera.release()
cv2.destroyAllWindows()
