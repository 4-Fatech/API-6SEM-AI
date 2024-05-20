import math
from ultralytics import YOLO
import cv2
from DatabaseHandler import DatabaseHandler


class PeopleCounter:
    def __init__(self, model_path="./runs/detect/train2/weights/best.pt", video_source='videoEdit.mp4', frame_skip=5):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_source)
        self.frame = None
        self.line_x = None
        self.pessoas_detectadas_entrando = dict()
        self.pessoas_detectadas_saindo = dict()
        self.lotacao_atual = dict()
        self.classNames = ["person"]
        self.db_handler = DatabaseHandler()
        self.frame_skip = frame_skip  # Número de frames a serem pulados entre processamentos

        # # Verificar se a GPU está disponível
        # self.cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        # if self.cuda_available:
        #     print("CUDA está disponível!")
        # else:
        #     print("CUDA não está disponível, utilizando CPU para processamento.")

    def run(self):
        list = self.db_handler.listRedzone()
        frame_count = 0  # Contador de frames

        for r in list.data:
            id = r["id_redzone"]
            self.lotacao_atual[id] = 0
            self.pessoas_detectadas_entrando[id] = []
            self.pessoas_detectadas_saindo[id] = []
        
        while True:
            for r in list.data:
                ret, frame = self.cap.read()
                id = r["id_redzone"]
                
                if not ret:
                    break

                frame_count += 1

                # Processa apenas se frame_count for múltiplo de frame_skip
                if frame_count % self.frame_skip != 0:
                    continue

                height, width, _ = frame.shape
                self.line_x = width // 2

                # Converter a imagem para escala de cinza usando a GPU (CUDA), se disponível
                # if self.cuda_available:
                #     gpu_frame = cv2.cuda_GpuMat()
                #     gpu_frame.upload(frame)
                #     gpu_gray_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
                #     gray_frame = gpu_gray_frame.download()
                # else:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
        acuracia = float(box.conf)

        if cls < len(class_names) and class_names[cls] == "person" and acuracia >= 0.3:
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            if x1 + w // 2 > self.line_x - 200 and x1 + w // 2 < self.line_x + 200:
                if (x1 + w // 2 < self.line_x and box.id not in self.pessoas_detectadas_entrando[redzone]):
                    self.pessoas_detectadas_entrando[redzone].append(box.id)
                if (x1 + w // 2 > self.line_x and box.id not in self.pessoas_detectadas_saindo[redzone]):
                    self.pessoas_detectadas_saindo[redzone].append(box.id)
                if x1 + w // 2 > self.line_x and box.id in self.pessoas_detectadas_entrando[redzone]:
                    self.pessoas_detectadas_entrando[redzone].remove(box.id)
                    self.lotacao_atual[redzone] += 1
                    self.db_handler.insert_record(True, self.lotacao_atual[redzone], redzone)
                if x1 + w // 2 < self.line_x and box.id in self.pessoas_detectadas_saindo[redzone]:
                    self.pessoas_detectadas_saindo[redzone].remove(box.id)
                    if self.lotacao_atual[redzone] != 0:
                        self.lotacao_atual[redzone] -= 1
                        self.db_handler.insert_record(False, self.lotacao_atual[redzone], redzone)
            text = f'Class: {class_names[cls]}, Acuracia: {acuracia:.2f}'
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def display(self, frame, redzone):
        if frame is not None:
            height, width, _ = frame.shape
            cv2.line(frame, (self.line_x, 0), (self.line_x, height), (0, 255, 0), 2)
            cv2.putText(frame, f"Redzone: {redzone}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("frame" + str(redzone), frame)


if __name__ == "__main__":
    counter = PeopleCounter(frame_skip=3)  # Ajuste frame_skip conforme necessário
    counter.run()
