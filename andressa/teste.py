import cv2
import numpy as np

def process_frame(frame):
    # Converter para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Faixas da cor vermelha no HSV
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Máscara para vermelho
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Redução de ruído
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:  # Ajuste conforme necessidade
            x, y, w, h = cv2.boundingRect(cnt)

            # Análise do formato da bounding box
            aspect_ratio = h / float(w) if w != 0 else 0

            

            # Considera como haste se for estreito e comprido
            if aspect_ratio > 2.5:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Haste ({aspect_ratio:.1f})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame, mask

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, mask = process_frame(frame)

        cv2.imshow("Deteccao de Haste", processed_frame)
        cv2.imshow("Mascara", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Salvar frame se necessário para debugging
            cv2.imwrite("frame_salvo.png", frame)
            cv2.imwrite("mascara_salva.png", mask)
            print("Frame salvo.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
