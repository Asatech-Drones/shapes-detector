import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Cria uma única janela nomeada antes do loop
cv2.namedWindow('Detecção de Círculos', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Sai do loop se a captura falhar

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
        param1=50, param2=30, minRadius=20, maxRadius=100
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            cv2.circle(frame, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
            cv2.circle(frame, (circle[0], circle[1]), 2, (0, 0, 255), 3)

    # Exibe o frame na janela criada anteriormente
    cv2.imshow('Detecção de Círculos', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
