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
        self.pessoas_detectadas_entrando = []
        self.pessoas_detectadas_saindo = []
        self.lotacao_atual = 0
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
        while True:
            ret, self.frame = self.cap.read()

            if not ret:
                break

            height, width, _ = self.frame.shape
            self.line_x = width // 2

            results = self.model.track(self.frame, persist=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    self.process_box(box, self.classNames)

            self.display()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def process_box(self, box, class_names):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1

        cv2.rectangle(self.frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

        cls = int(box.cls[0])
        # Lado esquerdo pro direito -> entra
        # Lado direito pro esquerdo -> sai
        if class_names[cls] == "person":
              if x1+w//2 > self.line_x-200 and x1+w//2 < self.line_x+200:
                if (x1 + w // 2 < self.line_x and box.id not in self.pessoas_detectadas_entrando):
                    #Lado esquerdo e o id nao ta no array de entrada -> adiciona
                    self.pessoas_detectadas_entrando.append(box.id)
                    
                if (x1 + w // 2 > self.line_x and box.id not in self.pessoas_detectadas_saindo):
                    #Lado direito e o id nao ta no array de saida -> adiciona
                    self.pessoas_detectadas_saindo.append(box.id)
                if x1 + w // 2 > self.line_x and box.id in self.pessoas_detectadas_entrando:
                    #Lado direito e o id esta no array de entrada -> remove
                    self.pessoas_detectadas_entrando.remove(box.id)
                    self.lotacao_atual += 1
                    self.db_handler.insert_record(True, self.lotacao_atual)
                if x1 + w // 2 < self.line_x and box.id in self.pessoas_detectadas_saindo:
                    #Lado esquerdo e o id esta no array de saida -> remove
                    self.pessoas_detectadas_saindo.remove(box.id)
                    if self.lotacao_atual != 0:
                        self.lotacao_atual -= 1
                        self.db_handler.insert_record(False, self.lotacao_atual)

    def display(self):
        if self.frame is not None:
            height, width, _ = self.frame.shape
            cv2.line(self.frame, (self.line_x, 0), (self.line_x, height), (0, 255, 0), 2)
            cv2.line(self.frame, (self.line_x-200, 0), (self.line_x-200, height), (0, 255, 0), 2)
            cv2.line(self.frame, (self.line_x+200, 0), (self.line_x+200, height), (0, 255, 0), 2)
            cv2.putText(self.frame,f"Lotacao Atual: {self.lotacao_atual}",(10, 50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2,)
            cv2.imshow("frame", self.frame)


if __name__ == "__main__":
    counter = PeopleCounter()
    counter.run()
