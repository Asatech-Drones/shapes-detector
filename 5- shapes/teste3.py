import cv2
import numpy as np

def identify_shape(approx):
    sides = len(approx)
    if sides == 3:
        return "Triangulo"
    elif sides == 4:
        return "Quadrado"
    elif sides == 5:
        return "Pentagono"
    elif sides == 6:
        return "Hexagono"
    elif sides > 6:
        return "Circulo"
    return "Desconhecido"

# Define as faixas de cores no espaço HSV (valores aproximados)
color_ranges = {
    "Circulo": ([100, 150, 50], [140, 255, 255]),  # Azul
    "Quadrado": ([0, 50, 0], [10, 255, 50]),  # Marrom escuro
    "Triangulo": ([90, 50, 50], [130, 255, 255]),  # Azul escuro
    "Hexagono": ([0, 100, 100], [10, 255, 255]),  # Vermelho
    "Pentagono": ([0, 100, 50], [10, 255, 100]),  # Vermelho escuro
    "Estrela": ([40, 100, 50], [80, 255, 255]),  # Verde
    "Cruz": ([140, 100, 100], [160, 255, 255]),  # Rosa
    "Casa": ([10, 100, 100], [20, 255, 255])  # Laranja
}

cap = cv2.VideoCapture(0)
# Cria uma única janela nomeada antes do loop
cv2.namedWindow('Detecção de Formas', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    for shape, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        
        # Encontra contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:  # Remove ruídos
                approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
                detected_shape = identify_shape(approx)
                
                if detected_shape == shape:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("Detecção de Formas", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()