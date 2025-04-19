import cv2
import numpy as np

filter_shapes = [
    "Circulo Azul",
    "Quadrado Bordo",
    "Retangulo Bordo",
    "Triangulo Azul Escuro",
    "Hexagono Vermelho",
    "Pentagono Marrom"
]

# Função para verificar se a cor está dentro de um intervalo
def get_color_range(hex_color):
    # Converter hex para RGB
    rgb = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]

    # Converter RGB para HSV
    hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_BGR2HSV)[0][0]

    # Tolerância para a variação da cor (aqui estamos permitindo ±10 para Hue e ±50 para Saturation e Value)
    lower_range = np.array(
        [hsv[0] - 10, max(0, hsv[1] - 50), max(0, hsv[2] - 50)], 
         dtype=np.uint8
    )
    upper_range = np.array(
        [hsv[0] + 10, min(255, hsv[1] + 50), min(255, hsv[2] + 50)], 
        dtype=np.uint8
    )
    
    return lower_range, upper_range

# Função para detectar cores na imagem
def figColor(imagenHSV):
    # Definir as cores hexadecimais para detectar
    color_hex_codes = [
        '0000ff', '2b0000', '002255', 'ff0000', 
        '800000', '008000', 'ff00ff', 'ff6600'
    ]
    
    # Verificar cada cor
    for color_hex in color_hex_codes:
        lower_range, upper_range = get_color_range(color_hex)
        
        # Criar a máscara para a faixa de cor
        mask = cv2.inRange(imagenHSV, lower_range, upper_range)
        
        # Verificar se há algum pixel dentro da faixa de cor
        if cv2.countNonZero(mask) > 0:
            return color_hex  # Retorna o código da cor detectada
    
    return '?'

def figName(contorno, width, height):
    epsilon = 0.01 * cv2.arcLength(contorno, True)
    approx = cv2.approxPolyDP(contorno, epsilon, True)

    if len(approx) == 3:
        return 'Triangulo'
    elif len(approx) == 4:
        aspect_ratio = float(width) / height
        return 'Quadrado' if 0.95 <= aspect_ratio <= 1.05 else 'Retangulo'
    elif len(approx) == 5:
        return 'Pentagono'
    elif len(approx) == 6:
        return 'Hexagono'
    elif len(approx) > 10:
        return 'Circulo'
    return '?'

# Inicializa a webcam
cap = cv2.VideoCapture(0)
# Cria uma única janela nomeada antes do loop
cv2.namedWindow('Detecção de Formas e Cores', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Conversão para tons de cinza e aplicação do Canny
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(gray, 100, 200)
    canny = cv2.dilate(canny, None, iterations=1)
    canny = cv2.erode(canny, None, iterations=1)
    
    # Detecção de contornos
    cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtra contornos pequenos
    # Ajuste conforme necessário
    min_area = 250
    cnts = [c for c in cnts if cv2.contourArea(c) > min_area]
    
    imageHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        imAux = np.zeros(frame.shape[:2], dtype="uint8")
        imAux = cv2.drawContours(imAux, [c], -1, 255, -1)
        maskHSV = cv2.bitwise_and(imageHSV, imageHSV, mask=imAux)

        name = figName(c, w, h)
        color = figColor(maskHSV)
        nameColor = name + ' ' + color

        # if(nameColor in filter_shapes):
        cv2.putText(frame, nameColor, (x, y - 5), 1, 0.8, (0, 255, 0), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Detecção de Formas e Cores', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
