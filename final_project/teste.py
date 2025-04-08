import cv2
import numpy as np

def process_frame(frame):
    # Converter o frame para o espaço de cor HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Definir os limites inferior e superior para a cor vermelha
    # Note que o vermelho pode ser segmentado em duas faixas no espaço HSV
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Criar as máscaras para as duas faixas
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Redução de ruído com operações morfológicas
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Encontrar contornos nas áreas detectadas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Percorrer cada contorno e desenhar retângulo se a área for significativa
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:  # filtro para descartar pequenas regiões de ruído
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Haste", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame, mask

def main():
    # Inicializa a captura de vídeo (0 para webcam ou substitua por outra fonte)
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Processa o frame para detectar a cor vermelha
        processed_frame, mask = process_frame(frame)

        # Exibe o frame processado e a máscara
        cv2.imshow("Deteccao de Haste", processed_frame)
        cv2.imshow("Mascara", mask)

        # Sai do loop quando 'q' é pressionado
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()