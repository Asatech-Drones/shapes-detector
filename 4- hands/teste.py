import cv2
import mediapipe as mp
import numpy as np

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Função para calcular a distância entre dois pontos
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Inicializa a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

# Cria uma única janela nomeada antes do loop
cv2.namedWindow('Medicao de Distancia', cv2.WINDOW_NORMAL)

# Distância focal da câmera (ajuste conforme necessário)
focal_length = 1000  # Exemplo: valor em pixels

# Tamanho real de um objeto de referência (em centímetros)
real_width = 10  # Exemplo: largura de um objeto conhecido

while True:
    # Captura frame por frame
    ret, frame = cap.read()
    if not ret:
        break

    # Converte o frame para RGB (MediaPipe requer RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processa o frame para detectar mãos
    results = hands.process(rgb_frame)

    # Se mãos forem detectadas
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Desenha os landmarks da mão
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Obtém as coordenadas dos dedos (por exemplo, ponta do polegar e do indicador)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Converte as coordenadas normalizadas para pixels
            h, w, _ = frame.shape
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

            # Desenha os pontos dos dedos
            cv2.circle(frame, (thumb_x, thumb_y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (index_x, index_y), 5, (0, 255, 0), -1)

            # Calcula a distância entre os pontos (em pixels)
            distance_pixels = calculate_distance((thumb_x, thumb_y), (index_x, index_y))

            # Converte a distância para centímetros (ajuste conforme necessário)
            distance_cm = (distance_pixels * real_width) / focal_length

            # Exibe a distância na tela
            cv2.putText(frame, f"Distancia: {distance_cm:.2f} cm", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Exibe o frame
    cv2.imshow('Medicao de Distancia', frame)

    # Aguarda 1ms e verifica se a tecla 'q' foi pressionada para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()