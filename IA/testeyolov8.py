import math
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

video_path="./pessoas.mp4"
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

cap = cv2.VideoCapture(0)

ret = True
# Lista para armazenar as pessoas detectadas
pessoas_detectadas_entrando = []
pessoas_detectadas_saindo = []

# Contador para a lotação atual da sala
lotacao_atual = 0

while ret:
    ret,frame = cap.read()

    height, width, _ = frame.shape
    line_x = width // 2
    

    results = model.track(frame,persist=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
 
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass ==  'person':
                
                cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)

                if x1+w//2 > line_x-200 and x1+w//2 < line_x+200:
                    if x1+w//2 < line_x and not pessoas_detectadas_entrando.__contains__(box.id):
                        pessoas_detectadas_entrando.append(box.id)
                    
                    if x1+w//2 >  line_x and not pessoas_detectadas_saindo.__contains__(box.id):
                        pessoas_detectadas_saindo.append(box.id)

                    if x1+w//2 > line_x and  pessoas_detectadas_entrando.__contains__(box.id):
                        pessoas_detectadas_entrando.remove(box.id)
                        lotacao_atual = lotacao_atual + 1
                    
                    if x1+w//2 <  line_x and  pessoas_detectadas_saindo.__contains__(box.id):
                        pessoas_detectadas_saindo.remove(box.id)
                        if lotacao_atual != 0:
                            lotacao_atual = lotacao_atual -1
                    
        


       
 
    cv2.line(frame, (line_x, 0), (line_x, height), (0, 255, 0), 2)
    cv2.line(frame, (line_x-200, 0), (line_x-200, height), (0, 255, 0), 2)
    cv2.line(frame, (line_x+200, 0), (line_x+200, height), (0, 255, 0), 2)
    cv2.putText(frame, f"Lotacao Atual: {lotacao_atual}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
