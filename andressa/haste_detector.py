import numpy as np
import cv2


class HasteDetector:
    def __init__(self):
        # Intervalos de cores para detectar as hastes
        self.color_ranges = {
        'red': [
            (np.array([0, 70, 50]), np.array([10, 255, 255])),
            (np.array([170, 70, 50]), np.array([180, 255, 255]))
        ],
        'orange': [
            (np.array([10, 100, 200]), np.array([20, 255, 255]))
        ],
        'yellow': [
            (np.array([8, 80, 130]), np.array([20, 255, 255]))
        ],
        'blue': [
            (np.array([100, 50, 50]), np.array([130, 255, 255]))
        ],
        'green': [
            (np.array([40, 70, 70]), np.array([80, 255, 255]))
        ]
    }
        self.last_detection = None  # Armazena a última detecção para comparação
        self.was_visible = False

    def detect_staves(self, frame, target_color, frame_width):
        """Detecta hastes coloridas no frame, com melhorias para fundo variado."""

        # 1. Pré-processamento: desfoque e conversão HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 2. Equaliza o canal V (iluminação)
        h, s, v = cv2.split(hsv)
        v = cv2.equalizeHist(v)
        hsv = cv2.merge((h, s, v))

        detections = []
        mask_total = np.zeros(frame.shape[:2], dtype=np.uint8)

        # 3. Aplica máscara com todas as faixas definidas para a cor
        for lower, upper in self.color_ranges.get(target_color, []):
            color_mask = cv2.inRange(hsv, lower, upper)
            mask_total = cv2.bitwise_or(mask_total, color_mask)

        # 4. Limpeza da máscara com morfologia
        kernel = np.ones((3, 3), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel, iterations=2)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 5. Detecção de contornos
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 350:  # área mínima
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = h / float(w) if w > 0 else 0
                if aspect_ratio > 2.0:  # vertical e fino
                    detections.append({'x': x, 'y': y, 'w': w, 'h': h})
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{target_color} Haste", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 6. Detecção mais próxima ao centro
        if detections:
            closest_haste = self.determine_closest_haste(detections, frame_width)
            if self.last_detection is None or self.is_similar_detection(closest_haste, self.last_detection):
                self.last_detection = closest_haste
            else:
                self.last_detection = None

        return frame, detections, mask_cleaned


    def determine_closest_haste(self, detections, frame_width):
        """Função para determinar a haste mais próxima do centro do frame."""
        if not detections:
            return None
        center_x = frame_width // 2
        detections.sort(key=lambda d: abs((d['x'] + d['w'] // 2) - center_x))
        # recebe uma lista de hastes detectadas
        # ordena pela proximidade do centro horizontal do frame
        # escolhe a mais próxima do centro da imagem
        return detections[0]

    def is_similar_detection(self, new_detection, last_detection):
        """Verifica se a nova detecção é similar à última detecção."""
        if not new_detection or not last_detection:
            return False

        # Tolerância para aceitar pequenas variações
        tolerance = 50
        x_diff = abs(new_detection['x'] - last_detection['x'])
        y_diff = abs(new_detection['y'] - last_detection['y'])
        w_diff = abs(new_detection['w'] - last_detection['w'])
        h_diff = abs(new_detection['h'] - last_detection['h'])

        # A detecção é considerada similar se a diferença nas coordenadas e tamanho for pequena
        return x_diff < tolerance and y_diff < tolerance and w_diff < tolerance and h_diff < tolerance
