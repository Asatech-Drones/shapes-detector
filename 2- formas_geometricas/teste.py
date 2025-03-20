import cv2
import numpy as np

# Inicializa a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

# Cria uma única janela nomeada antes do loop
cv2.namedWindow('Detecção de Formas Geométricas', cv2.WINDOW_NORMAL)

while True:
    # Captura frame por frame
    ret, frame = cap.read()

    if not ret:
      break  # Sai do loop se a captura falhar

    # Converte o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplica um desfoque para reduzir ruídos
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binariza a imagem usando o método de Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Encontra contornos na imagem
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Itera sobre os contornos encontrados
    for contour in contours:
        # Aproxima o contorno para uma forma geométrica
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Desenha o contorno e a forma aproximada
        cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)

        # Identifica a forma com base no número de vértices
        vertices = len(approx)
        shape_name = ""
        if vertices == 3:
            shape_name = "Triângulo"
        elif vertices == 4:
            # Verifica se é um quadrado ou retângulo
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            shape_name = "Quadrado" if 0.95 <= aspect_ratio <= 1.05 else "Retângulo"
        elif vertices == 5:
            shape_name = "Pentágono"
        elif vertices == 6:
            shape_name = "Hexágono"
        else:
            shape_name = "Círculo" if vertices > 6 else "Desconhecido"

        # Adiciona o nome da forma ao frame
        cv2.putText(frame, shape_name, (approx.ravel()[0], approx.ravel()[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Exibe o frame com as formas detectadas
    cv2.imshow('Detecção de Formas Geométricas', frame)

    # Aguarda 1ms e verifica se a tecla 'q' foi pressionada para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()