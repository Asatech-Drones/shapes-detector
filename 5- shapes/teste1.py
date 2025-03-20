import cv2
import numpy as np
import os

def load_templates(template_folder):
    """Carrega todas as imagens de referência do diretório especificado."""
    templates = {}
    for filename in os.listdir(template_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # Remove a extensão do arquivo
            shape_name = os.path.splitext(filename)[0] 
            templates[shape_name] = cv2.imread(
                os.path.join(template_folder, filename), 0
            )
    return templates

def detect_shapes(frame, templates):
    """Detecta formas geométricas no frame usando template matching."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_shapes = []
    
    for shape, template in templates.items():
        h, w = template.shape
        res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.6  # Ajuste conforme necessário
        loc = np.where(res >= threshold)
        
        for pt in zip(*loc[::-1]):
            detected_shapes.append((shape, pt[0], pt[1], w, h))
            cv2.rectangle(
                frame, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), (0, 255, 0), 2
            )
            cv2.putText(
                frame, shape, (pt[0], pt[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 255, 0), 2
            )
    
    return detected_shapes

def determine_position(x, y, frame_width, frame_height):
    """Determina a posição do objeto na imagem."""
    if x < frame_width / 3:
        position_x = "Esquerda"
    elif x > 2 * frame_width / 3:
        position_x = "Direita"
    else:
        position_x = "Centro"
    
    if y < frame_height / 3:
        position_y = "Topo"
    elif y > 2 * frame_height / 3:
        position_y = "Baixo"
    else:
        position_y = "Centro"
    
    return f"{position_y} - {position_x}"

# Caminho para a pasta onde estão as imagens de referência
template_folder = "templates"

# Carregar templates
templates = load_templates(template_folder)

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(0)

# Cria uma única janela nomeada antes do loop
cv2.namedWindow('Detecção de Formas', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    detected_shapes = detect_shapes(frame, templates)
    frame_height, frame_width = frame.shape[:2]
    
    for shape, x, y, w, h in detected_shapes:
        position = determine_position(x, y, frame_width, frame_height)
        cv2.putText(
            frame, position, 
            (x, y + h + 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, (0, 255, 255), 2
        )
    
    cv2.imshow("Detecção de Formas", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()