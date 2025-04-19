import cv2
import numpy as np

def detect_shapes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Aplicar detecção de bordas
    edges = cv2.Canny(blurred, 50, 150)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image.shape[:2]
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        
        # Determinar a posição na imagem
        center_x, center_y = x + w // 2, y + h // 2
        position = ""
        if center_x < width * 0.33:
            position = "Esquerda"
        elif center_x > width * 0.66:
            position = "Direita"
        else:
            position = "Centro"
        
        if center_y < height * 0.33:
            position += " - Superior"
        elif center_y > height * 0.66:
            position += " - Inferior"
        else:
            position += " - Meio"
        
        # Identificar a forma
        shape = "Desconhecido"
        num_sides = len(approx)
        if num_sides == 3:
            shape = "Triângulo"
        elif num_sides == 4:
            aspect_ratio = w / float(h)
            if 0.9 < aspect_ratio < 1.1:
                shape = "Quadrado"
            else:
                shape = "Retângulo"
        elif num_sides == 5:
            shape = "Pentágono"
        elif num_sides == 6:
            shape = "Hexágono"
        elif num_sides > 6:
            shape = "Círculo"
        
        # Desenhar e mostrar informações na imagem
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
        cv2.putText(
            image, f"{shape}, {position}", 
            (x, y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, (255, 255, 255), 2
        )
    
    return image

# Captura da webcam
cap = cv2.VideoCapture(0)

# Cria uma única janela nomeada antes do loop
cv2.namedWindow('Shapes', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    result = detect_shapes(frame)
    cv2.imshow("Shapes", result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()