import cv2
import numpy as np
import glob

# Definição das cores esperadas para cada forma (em HSV)
colors = {
    "circulo": ((100, 150, 50), (140, 255, 255)),  # Azul
    "quadrado": ((0, 50, 0), (10, 255, 50)),       # Marrom escuro
    "triangulo": ((90, 50, 50), (130, 255, 150)),  # Azul escuro
    "hexagono": ((0, 150, 150), (10, 255, 255)),   # Vermelho
    "pentagono": ((0, 100, 100), (10, 255, 200)),  # Vermelho escuro
    "estrela": ((50, 100, 50), (70, 255, 255)),    # Verde
    "cruz": ((140, 100, 100), (160, 255, 255)),    # Rosa
    "casa": ((10, 150, 150), (20, 255, 255)),      # Laranja
}

# Carregar templates das formas geométricas
templates = {}
orb = cv2.ORB_create(nfeatures=500)

for template_path in glob.glob("templates/*.png"):
    # Nome do arquivo sem extensão
    name = template_path.split("/")[-1].split(".")[0]
    img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    kp, des = orb.detectAndCompute(img, None)
    templates[name] = (kp, des, img)

# Iniciar captura de vídeo
cap = cv2.VideoCapture(0)
flann = cv2.FlannBasedMatcher(
    dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1), 
    {}
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp_scene, des_scene = orb.detectAndCompute(gray, None)

    for shape_name, (kp_template, des_template, template_img) in templates.items():
        if des_scene is None or des_template is None:
            continue

        matches = flann.knnMatch(des_template, des_scene, k=2)
        good_matches = [m for match in matches if len(match) > 1 for m, n in [match] if m.distance < 0.7 * n.distance]


        if len(good_matches) > 10:
            src_pts = np.float32(
              [kp_template[m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
              [kp_scene[m.trainIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                h, w = template_img.shape
                pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                # Verifica a cor na região detectada
                x_min, y_min = np.int32(dst.min(axis=0)).flatten()
                x_max, y_max = np.int32(dst.max(axis=0)).flatten()

                if 0 <= x_min < frame.shape[1] and 0 <= y_min < frame.shape[0] and \
                   0 <= x_max < frame.shape[1] and 0 <= y_max < frame.shape[0]:

                    roi = hsv[y_min:y_max, x_min:x_max]
                    lower, upper = colors[shape_name]
                    mask = cv2.inRange(roi, lower, upper)

                    # Certifica que a cor corresponde
                    if np.sum(mask) > 1000:  
                        cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3)
                        cv2.putText(
                            frame, shape_name, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                        )

    cv2.imshow("Shape Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
