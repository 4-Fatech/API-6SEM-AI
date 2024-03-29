import cv2
import numpy as np  

# Carregar o modelo YOLOv4 e as classes de objetos
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
classes = []
with open("coco.names", 'r') as f:
    classes = f.read().rstrip('\n').split('\n')

# Capturar a entrada de vídeo
cap = cv2.VideoCapture(0)

# Dimensões da imagem
_, img = cap.read()
height, width, _ = img.shape

# Coordenadas da linha central
line_x = width // 2

# Lista para armazenar as pessoas detectadas
pessoas_detectadas = []

# Contador para a lotação atual da sala
lotacao_atual = 0

while True:
    # Capturar frame por frame
    sucesso, img = cap.read()

    # Se a câmera não abrir
    if not sucesso:
        print("Erro ao abrir câmera")
        break

    # Desenhar a linha central
    cv2.line(img, (line_x, 0), (line_x, height), (0, 255, 0), 2)

    # Obter as dimensões do frame
    height, width, _ = img.shape

    # Normalizar a entrada, redimensionando para 416x416 e dividindo por 255
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Configurar a entrada para a rede
    net.setInput(blob)

    # Executar a detecção
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Limpar a lista de pessoas detectadas para cada iteração
    pessoas_detectadas = []

    # Processar as detecções
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Se a confiança for alta o suficiente e for uma pessoa
                # Obter as coordenadas do retângulo
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Adicionar a pessoa à lista de pessoas detectadas
                pessoas_detectadas.append((x, y, w, h))

                # Desenhar um retângulo em torno da pessoa detectada
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Determinar a lotação atual da sala
    lotacao_atual = len(pessoas_detectadas)

    # Exibir a lotação atual da sala
    cv2.putText(img, f"Lotacao Atual: {lotacao_atual}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Exibir o frame com as pessoas detectadas
    cv2.imshow("Pessoas Detectadas", img)

    # Definir a tecla ou ação para parar o looping
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fechar a janela e liberar a câmera
cap.release()
cv2.destroyAllWindows()
