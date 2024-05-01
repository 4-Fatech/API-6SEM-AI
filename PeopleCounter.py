import math
from ultralytics import YOLO
import cv2
from DatabaseHandler import DatabaseHandler

class PeopleCounter:
    def __init__(self, model_path="yolov8n.pt", video_source=0):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_source)
        self.frame = None
        self.line_x = None
        self.pessoas_detectadas_entrando = dict()
        self.pessoas_detectadas_saindo = dict()
        self.lotacao_atual = dict()
        self.classNames = ["person","bicycle","car","motorbike",
            "aeroplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign",
            "parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
            "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
            "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
            "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
            "broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed",
            "diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave",
            "oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
        ]

        self.db_handler = DatabaseHandler()

    def run(self):
        list = self.db_handler.listRedzone()
        for r in list.data:
                id = r["id_redzone"]
                self.lotacao_atual[id] = 0
                self.pessoas_detectadas_entrando[id]=[]
                self.pessoas_detectadas_saindo[id]=[]
        while True:
            for r in list.data:
                ret, frame = self.cap.read()
                id = r["id_redzone"]

             
                if not ret:
                    break

                height, width, _ = frame.shape
                self.line_x = width // 2

                results = self.model.track(frame, persist=True)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        self.process_box(box, self.classNames, frame, redzone=id)

                self.display(frame, id)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        self.cap.release()
        cv2.destroyAllWindows()

    def process_box(self, box, class_names, frame, redzone):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1

       

        cls = int(box.cls[0])
        # Lado esquerdo pro direito -> entra
        # Lado direito pro esquerdo -> sai

        # Acuracia
        acuracia = float(box.conf) 

        # Pegando box ID
        if box.id is not None and box.id.item() is not None:  
            boxId = int(box.id.item())  
            
        if class_names[cls] == "person" and acuracia >= 0.8:         
            
             # Exibir ID e Acurácia
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            if x1+w//2 > self.line_x-200 and x1+w//2 < self.line_x+200:

                if (x1 + w // 2 < self.line_x and box.id not in self.pessoas_detectadas_entrando[redzone]):
                    #Lado esquerdo e o id nao ta no array de entrada -> adiciona
                    self.pessoas_detectadas_entrando[redzone].append(box.id)
                    
                if (x1 + w // 2 > self.line_x and box.id not in self.pessoas_detectadas_saindo[redzone]):
                    #Lado direito e o id nao ta no array de saida -> adiciona
                    self.pessoas_detectadas_saindo[redzone].append(box.id)
                if x1 + w // 2 > self.line_x and box.id in self.pessoas_detectadas_entrando[redzone]:
                    #Lado direito e o id esta no array de entrada -> remove
                    self.pessoas_detectadas_entrando[redzone].remove(box.id)
                    self.lotacao_atual[redzone] += 1

                    self.db_handler.insert_record(True, self.lotacao_atual[redzone], redzone)

                if x1 + w // 2 < self.line_x and box.id in self.pessoas_detectadas_saindo[redzone]:
                    #Lado esquerdo e o id esta no array de saida -> remove
                    self.pessoas_detectadas_saindo[redzone].remove(box.id)
                    if self.lotacao_atual[redzone] != 0:
                        self.lotacao_atual[redzone] -= 1
                        self.db_handler.insert_record(False, self.lotacao_atual[redzone],redzone)
            text = f'Class: {class_names[cls]}, ID: {int(box.id.item())}, Acuracia: {acuracia:.2f}'
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



    def display(self, frame, redzone):
        if frame is not None:
            height, width, _ = frame.shape
            cv2.line(frame, (self.line_x, 0), (self.line_x, height), (0, 255, 0), 2)
            cv2.line(frame, (self.line_x-200, 0), (self.line_x-200, height), (0, 255, 0), 2)
            cv2.line(frame, (self.line_x+200, 0), (self.line_x+200, height), (0, 255, 0), 2)
            cv2.putText(frame,f"Lotacao Atual: {self.lotacao_atual[redzone]}",(10, 50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2,)
            cv2.imshow("frame"+str(redzone), frame)


if __name__ == "__main__":
    counter = PeopleCounter()
    counter.run()
