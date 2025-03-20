import cv2

# Inicializa a captura de vídeo da webcam (0 é o índice da câmera padrão)
cap = cv2.VideoCapture(0)

while True:
    # Captura frame por frame
    ret, frame = cap.read()

    # Exibe o frame em uma janela
    cv2.imshow('Webcam', frame)

    # Aguarda 1ms e verifica se a tecla 'q' foi pressionada para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()