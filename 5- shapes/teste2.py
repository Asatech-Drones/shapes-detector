import cv2
import numpy as np
import os

# Caminho dos templates
TEMPLATE_PATH = "templates/"

# Carrega os templates e associa a um nome
templates = {}
for filename in os.listdir(TEMPLATE_PATH):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        # Remove extensão
        nome = filename.split(".")[0]  
        # Converte para escala de cinza
        templates[nome] = cv2.imread(os.path.join(TEMPLATE_PATH, filename), 0)

# Inicializa a captura da webcam
cap = cv2.VideoCapture(0)
# Cria uma única janela nomeada antes do loop
cv2.namedWindow('Detecção de Formas', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Reduz o tamanho da imagem para melhorar a performance
    # Reduz para 50% do tamanho original
    scale_percent = 50
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    frame_resized = cv2.resize(frame, (width, height))

    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    detected_shapes = []

    # Percorre os templates para detectar as formas
    for shape_name, template in templates.items():
        # Reduz o template para melhor performance
        template = cv2.resize(
          template, (template.shape[1] // 2, template.shape[0] // 2)
        ) 
        w, h = template.shape[::-1]

        # Aplicando template matching
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        # Ajuste fino do limite de detecção
        threshold = 0.6  
        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            detected_shapes.append((pt[0], pt[1], w, h, shape_name))

    # Agrupar detecções sobrepostas
    if detected_shapes:
        rects = np.array([[x, y, x + w, y + h] for x, y, w, h, _ in detected_shapes])

        if len(rects) > 0:
            grouped_rects, _ = cv2.groupRectangles(
                rects.tolist(), groupThreshold=1, eps=0.5
            )

            # Evita acessar índices inválidos
            for i in range(len(grouped_rects)):
                x, y, x2, y2 = grouped_rects[i]
                shape_name = detected_shapes[i][4] if i < len(detected_shapes) else "Desconhecido"

                # Desenha retângulo
                cv2.rectangle(frame_resized, (x, y), (x2, y2), (0, 255, 0), 2)
                # Adiciona nome da forma
                cv2.putText(
                    frame_resized, shape_name, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 2
                )

    # Exibe o frame processado
    cv2.imshow('Detecção de Formas', frame_resized)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura e fecha as janelas
cap.release()
cv2.destroyAllWindows()
