import cv2
import numpy as np
import time
import math
import threading
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from threading import Thread, Lock
from filterpy.kalman import KalmanFilter
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import random


# ---------- PARTE 1: DETECTOR DE HASTES ------------
class DetectorHastes:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        self.running = False
        self.frame = None
        self.hastes_detectadas = []  # Lista de hastes detectadas [(x, y, w, h, aspect_ratio, cor)]
        self.lock = Lock()  # Para acesso seguro aos dados compartilhados

        # Definir faixas de cores HSV para cada cor
        self.faixas_cores = {
            "vermelho": [
                (np.array([0, 70, 50]), np.array([10, 255, 255])),
                (np.array([170, 70, 50]), np.array([180, 255, 255]))
            ],
            "azul": [
                (np.array([100, 50, 50]), np.array([130, 255, 255]))
            ],
            "rosa": [
                (np.array([140, 50, 50]), np.array([170, 255, 255]))
            ],
            "preto": [
                (np.array([0, 0, 0]), np.array([180, 50, 50]))
            ]
        }

        # Cores BGR para visualização
        self.cores_bgr = {
            "vermelho": (0, 0, 255),
            "azul": (153, 82, 19),  # #135299
            "rosa": (155, 177, 240),  # #f0b19b
            "preto": (26, 26, 26)  # #1a1a1a
        }

    def iniciar(self):
        """Inicia a captura de vídeo em uma thread separada"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print("Erro ao abrir a câmera!")
            return False

        self.running = True
        self.thread = Thread(target=self._processar_video)
        self.thread.daemon = True
        self.thread.start()
        return True

    def parar(self):
        """Para a captura de vídeo"""
        self.running = False
        if hasattr(self, 'thread') and self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()

    def _processar_video(self):
        """Thread principal para processar o vídeo"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Erro ao ler frame da câmera!")
                break

            # Processar o frame para detectar hastes
            processed_frame, hastes = self._process_frame(frame)

            # Atualizar dados compartilhados com lock para evitar race conditions
            with self.lock:
                self.frame = processed_frame
                self.hastes_detectadas = hastes

            # Exibir o frame processado (opcional, para debugging)
            cv2.imshow("Detecção de Hastes", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def _process_frame(self, frame):
        """Processa um frame para detectar hastes de todas as cores"""
        # Converter para HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        hastes = []

        # Para cada cor, detectar hastes
        for cor_nome, faixas in self.faixas_cores.items():
            # Criar máscara combinando todas as faixas para esta cor
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

            for lower, upper in faixas:
                mask_temp = cv2.inRange(hsv, lower, upper)
                mask = cv2.bitwise_or(mask, mask_temp)

            # Redução de ruído
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

            # Contornos
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Cor para visualização
            cor_bgr = self.cores_bgr.get(cor_nome, (0, 255, 0))

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 200:  # Ajuste conforme necessidade
                    x, y, w, h = cv2.boundingRect(cnt)

                    # Análise do formato da bounding box
                    aspect_ratio = h / float(w) if w != 0 else 0

                    # Considera como haste se for estreito e comprido
                    if aspect_ratio > 2.5:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), cor_bgr, 2)
                        cv2.putText(frame, f"{cor_nome} ({aspect_ratio:.1f})", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_bgr, 2)

                        # Adicionar à lista de hastes detectadas
                        centro_x = x + w / 2
                        centro_y = y + h / 2
                        hastes.append((centro_x, centro_y, w, h, aspect_ratio, cor_nome))

        return frame, hastes

    def obter_hastes(self):
        """Retorna a lista atual de hastes detectadas"""
        with self.lock:
            return self.hastes_detectadas.copy()

    def obter_frame(self):
        """Retorna o frame atual processado"""
        with self.lock:
            if self.frame is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            return self.frame.copy()


# ---------- PARTE 2: CONVERSOR DE COORDENADAS ------------
class ConversorCoordenadas:
    def __init__(self, altura_camera=1.5, fov_horizontal=60, fov_vertical=45,
                 resolucao=(640, 480), matriz_calibracao=None):
        """
        Inicializa o conversor de coordenadas

        Parâmetros:
        - altura_camera: altura da câmera em metros
        - fov_horizontal: campo de visão horizontal em graus
        - fov_vertical: campo de visão vertical em graus
        - resolucao: resolução da câmera (largura, altura)
        - matriz_calibracao: matriz de calibração da câmera (opcional)
        """
        self.altura_camera = altura_camera
        self.fov_h = math.radians(fov_horizontal)
        self.fov_v = math.radians(fov_vertical)
        self.resolucao = resolucao
        self.matriz_calibracao = matriz_calibracao

        # Calcular fatores de escala (pixels por metro)
        self.escala_h = resolucao[0] / (2 * altura_camera * math.tan(self.fov_h / 2))
        self.escala_v = resolucao[1] / (2 * altura_camera * math.tan(self.fov_v / 2))

    def pixel_para_mundo(self, pixel_x, pixel_y, altura_objeto=0):
        """
        Converte coordenadas de pixel para coordenadas do mundo

        Parâmetros:
        - pixel_x, pixel_y: coordenadas do pixel na imagem
        - altura_objeto: altura do objeto em metros (padrão: 0, nível do solo)

        Retorna:
        - (x, y, z): coordenadas 3D no mundo em metros
        """
        # Centralizar as coordenadas do pixel
        centro_x = self.resolucao[0] / 2
        centro_y = self.resolucao[1] / 2

        # Calcular deslocamento em pixels do centro
        dx_pixel = pixel_x - centro_x
        dy_pixel = pixel_y - centro_y

        # Calcular ângulos
        angulo_h = math.atan2(dx_pixel, self.escala_h)
        angulo_v = math.atan2(dy_pixel, self.escala_v)

        # Calcular distância no plano horizontal
        altura_relativa = self.altura_camera - altura_objeto
        distancia = altura_relativa / math.tan(math.pi / 2 - angulo_v)

        # Calcular coordenadas x, y no mundo
        x = distancia * math.sin(angulo_h)
        y = distancia * math.cos(angulo_h)

        # Coordenadas relativas à posição da câmera
        return (x, y, altura_objeto)


# ---------- PARTE 3: FILTRO DE KALMAN PARA RASTREAMENTO ------------
class RastreadorKalman:
    def __init__(self, dt=0.1):
        self.filtros = {}  # Dicionário de filtros Kalman para cada haste
        self.dt = dt
        self.ultimo_id = 0
        self.max_idade = 10  # Número máximo de frames sem detecção antes de remover
        self.idades = {}  # Idade de cada rastreador

    def inicializar_filtro(self):
        """Inicializa um novo filtro de Kalman"""
        kf = KalmanFilter(dim_x=6, dim_z=3)  # Estado: [x, y, z, vx, vy, vz], Medição: [x, y, z]

        # Matriz de transição de estado (modelo de velocidade constante)
        kf.F = np.array([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0],
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Matriz de medição
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

        # Covariância do ruído de processo
        kf.Q = np.eye(6) * 0.01

        # Covariância do ruído de medição
        kf.R = np.eye(3) * 0.1

        # Covariância inicial de estado
        kf.P = np.eye(6) * 1.0

        return kf

    def atualizar(self, deteccoes):
        """
        Atualiza os rastreadores com novas detecções

        Parâmetros:
        - deteccoes: lista de detecções [(x, y, z), ...]

        Retorna:
        - posicoes_filtradas: lista de posições filtradas [(id, x, y, z), ...]
        """
        # Incrementar idade de todos os rastreadores
        for id_rastreador in list(self.idades.keys()):
            self.idades[id_rastreador] += 1

            # Remover rastreadores muito antigos
            if self.idades[id_rastreador] > self.max_idade:
                del self.filtros[id_rastreador]
                del self.idades[id_rastreador]

        # Associar detecções a rastreadores existentes
        deteccoes_associadas = set()
        for id_rastreador, kf in list(self.filtros.items()):
            if not deteccoes:  # Se não há detecções, apenas prever
                kf.predict()
                continue

            # Encontrar a detecção mais próxima
            estado_atual = kf.x[:3]
            min_dist = float('inf')
            melhor_deteccao_idx = -1

            for i, deteccao in enumerate(deteccoes):
                if i in deteccoes_associadas:
                    continue

                dist = np.linalg.norm(np.array(deteccao) - estado_atual)
                if dist < min_dist:
                    min_dist = dist
                    melhor_deteccao_idx = i

            # Se encontrou uma detecção próxima o suficiente
            if melhor_deteccao_idx >= 0 and min_dist < 2.0:  # Limiar de distância
                # Marcar como associada
                deteccoes_associadas.add(melhor_deteccao_idx)

                # Atualizar o filtro
                kf.predict()
                kf.update(deteccoes[melhor_deteccao_idx])

                # Resetar idade
                self.idades[id_rastreador] = 0
            else:
                # Apenas prever, sem atualização
                kf.predict()

        # Criar novos rastreadores para detecções não associadas
        for i, deteccao in enumerate(deteccoes):
            if i not in deteccoes_associadas:
                novo_id = self.ultimo_id + 1
                self.ultimo_id = novo_id

                # Inicializar novo filtro
                novo_filtro = self.inicializar_filtro()

                # Definir estado inicial
                novo_filtro.x = np.array([
                    deteccao[0], deteccao[1], deteccao[2], 0, 0, 0
                ]).reshape(6, 1)

                # Adicionar ao dicionário
                self.filtros[novo_id] = novo_filtro
                self.idades[novo_id] = 0

        # Obter posições filtradas
        posicoes_filtradas = []
        for id_rastreador, kf in self.filtros.items():
            x, y, z = kf.x[:3, 0]
            posicoes_filtradas.append((id_rastreador, x, y, z))

        return posicoes_filtradas


# ---------- PARTE 4: CLASSIFICADOR DE CORES ------------
class ClassificadorCores:
    def __init__(self):
        # Cores específicas (em BGR)
        self.cores_especificas = {
            "preto": (26, 26, 26),  # #1a1a1a (RGB: 26, 26, 26)
            "azul": (153, 82, 19),  # #135299 (RGB: 19, 82, 153)
            "rosa": (155, 177, 240),  # #f0b19b (RGB: 240, 177, 155)
            "vermelho": (0, 0, 255)  # Vermelho puro
        }

        # Treinar classificador
        self.scaler, self.clf = self._treinar_classificador()

    def _treinar_classificador(self):
        """Treina um classificador de cores usando SVM com as cores específicas"""
        # Dados de treinamento baseados nas cores específicas
        X_train = []
        y_train = []

        # Para cada cor específica, criar várias amostras com pequenas variações
        for idx, (nome_cor, (b, g, r)) in enumerate(self.cores_especificas.items()):
            # Converter para HSV
            hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
            h, s, v = hsv

            # Normalizar H para [0, 360)
            h = (int(h) * 2) % 360

            # Normalizar S e V para [0, 1]
            s = s / 255.0
            v = v / 255.0

            # Adicionar a cor exata
            X_train.append([b, g, r, h, s, v])
            y_train.append(idx)

            # Adicionar variações
            for _ in range(5):
                # Variação em BGR
                b_var = max(0, min(255, b + np.random.randint(-20, 21)))
                g_var = max(0, min(255, g + np.random.randint(-20, 21)))
                r_var = max(0, min(255, r + np.random.randint(-20, 21)))

                # Converter para HSV
                hsv_var = cv2.cvtColor(np.uint8([[[b_var, g_var, r_var]]]), cv2.COLOR_BGR2HSV)[0][0]
                h_var, s_var, v_var = hsv_var

                # Normalizar
                h_var = (int(h_var) * 2) % 360
                s_var = s_var / 255.0
                v_var = v_var / 255.0

                X_train.append([b_var, g_var, r_var, h_var, s_var, v_var])
                y_train.append(idx)

        # Converter para arrays numpy
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Normalizar dados
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Treinar classificador SVM
        clf = SVC(kernel='rbf', probability=True)
        clf.fit(X_train_scaled, y_train)

        return scaler, clf

    def classificar(self, b, g, r, h, s, v):
        """Classifica uma cor com base em suas características"""
        X_new = np.array([[b, g, r, h, s, v]])
        X_new_scaled = self.scaler.transform(X_new)
        pred = self.clf.predict(X_new_scaled)[0]
        cores = ["preto", "azul", "rosa", "vermelho"]
        return cores[pred]

    def extrair_caracteristicas(self, roi):
        """Extrai características de cor de uma região de interesse"""
        if roi.size == 0:
            return None

        # Calcular médias BGR
        b, g, r = cv2.mean(roi)[:3]

        # Converter para HSV e calcular médias
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.mean(hsv_roi)[:3]

        # Normalizar H para [0, 360)
        h = (h * 2) % 360

        # Normalizar S e V para [0, 1]
        s = s / 255.0
        v = v / 255.0

        return b, g, r, h, s, v


# ---------- PARTE 5: INTEGRADOR VISÃO-ROTA ------------
class IntegradorVisaoRota:
    def __init__(self, detector, conversor, raio_hastes=0.05):
        self.detector = detector
        self.conversor = conversor
        self.raio_hastes = raio_hastes
        self.posicoes_hastes = []  # Lista de posições 3D das hastes [(x, y, z, raio)]
        self.cores_hastes = []  # Lista de cores das hastes
        self.rastreador = RastreadorKalman()
        self.classificador = ClassificadorCores()
        self.mapa_cores = {
            "vermelho": (0, 0, 255),
            "azul": (153, 82, 19),  # #135299
            "rosa": (155, 177, 240),  # #f0b19b
            "preto": (26, 26, 26)  # #1a1a1a
        }

    def atualizar_posicoes(self, posicao_drone):
        """
        Atualiza as posições das hastes com base na detecção atual

        Parâmetros:
        - posicao_drone: posição atual do drone (x, y, z)
        """
        hastes_detectadas = self.detector.obter_hastes()

        # Converter detecções para coordenadas 3D
        deteccoes_3d = []
        self.cores_hastes = []  # Limpar cores anteriores

        for haste in hastes_detectadas:
            if len(haste) >= 6:  # Verificar se temos a cor na tupla
                pixel_x, pixel_y, largura, altura, aspect_ratio, cor = haste
                self.cores_hastes.append(cor)
            else:
                pixel_x, pixel_y, largura, altura, aspect_ratio = haste
                # Classificar cor usando o classificador
                frame = self.detector.obter_frame()
                x1, y1 = int(pixel_x - largura / 2), int(pixel_y - altura / 2)
                x2, y2 = int(pixel_x + largura / 2), int(pixel_y + altura / 2)

                # Garantir que esteja dentro dos limites da imagem
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1] - 1, x2)
                y2 = min(frame.shape[0] - 1, y2)

                # Extrair região de interesse
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    caracteristicas = self.classificador.extrair_caracteristicas(roi)
                    if caracteristicas:
                        cor = self.classificador.classificar(*caracteristicas)
                    else:
                        cor = "desconhecido"
                else:
                    cor = "desconhecido"

                self.cores_hastes.append(cor)

            # Converter de coordenadas de pixel para coordenadas do mundo
            x_rel, y_rel, z = self.conversor.pixel_para_mundo(pixel_x, pixel_y)

            # Converter para coordenadas globais (relativas à origem do mapa)
            x_global = posicao_drone[0] + x_rel
            y_global = posicao_drone[1] + y_rel

            deteccoes_3d.append((x_global, y_global, z))

        # Atualizar rastreadores Kalman
        posicoes_filtradas = self.rastreador.atualizar(deteccoes_3d)

        # Atualizar lista de posições
        self.posicoes_hastes = []
        for _, x, y, z in posicoes_filtradas:
            # Estimar o raio (simplificado)
            self.posicoes_hastes.append((x, y, z, self.raio_hastes))

        return self.posicoes_hastes

    def classificar_cores(self):
        """
        Classifica as cores das hastes com base na imagem atual
        """
        # Se já temos cores das hastes do detector, não precisamos reclassificar
        if len(self.cores_hastes) == len(self.posicoes_hastes):
            return self.cores_hastes

        frame = self.detector.obter_frame()
        if frame is None or not self.detector.obter_hastes():
            return []

        cores = []
        for haste in self.detector.obter_hastes():
            if len(haste) >= 6:  # Se já temos a cor na tupla
                _, _, _, _, _, cor = haste
                cores.append(cor)
                continue

            pixel_x, pixel_y, w, h, _ = haste

            # Extrair região da haste
            x1, y1 = int(pixel_x - w / 2), int(pixel_y - h / 2)
            x2, y2 = int(pixel_x + w / 2), int(pixel_y + h / 2)

            # Garantir que esteja dentro dos limites da imagem
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1] - 1, x2)
            y2 = min(frame.shape[0] - 1, y2)

            # Extrair região de interesse
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                cores.append("desconhecido")
                continue

            # Extrair características de cor
            caracteristicas = self.classificador.extrair_caracteristicas(roi)
            if caracteristicas is None:
                cores.append("desconhecido")
                continue

            # Classificar cor
            cor = self.classificador.classificar(*caracteristicas)
            cores.append(cor)

        self.cores_hastes = cores
        return cores

    def obter_obstaculos(self):
        """
        Retorna a lista de obstáculos para o planejador de rota
        """
        return self.posicoes_hastes

    def obter_traves_coloridas(self):
        """
        Retorna a lista de traves com suas cores para o planejador de rota
        """
        traves = []
        for i, (x, y, z, raio) in enumerate(self.posicoes_hastes):
            cor = self.cores_hastes[i] if i < len(self.cores_hastes) else "desconhecido"
            traves.append({"x": x, "y": y, "cor": cor})
        return traves


# ---------- PARTE 6: FUNÇÕES RRT* ------------
class Node:
    def __init__(self, point):
        self.point = point
        self.parent = None
        self.cost = 0.0


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def steer(from_point, to_point, step_size):
    if distance(from_point, to_point) < step_size:
        return to_point
    direction = np.array(to_point) - np.array(from_point)
    direction = direction / np.linalg.norm(direction)
    new_point = np.array(from_point) + step_size * direction
    return tuple(new_point)


def get_nearest(nodes, point):
    return min(nodes, key=lambda node: distance(node.point, point))


def get_nearby_nodes(nodes, new_node, radius):
    return sorted([node for node in nodes if distance(node.point, new_node.point) < radius],
                  key=lambda node: node.cost)


def rewire(nodes, new_node, nearby_nodes):
    for near_node in nearby_nodes:
        cost = new_node.cost + distance(new_node.point, near_node.point)
        if cost < near_node.cost:
            near_node.parent = new_node
            near_node.cost = cost
            nearby = get_nearby_nodes(nodes, new_node, radius=2.0)
            rewire(nodes, new_node, nearby)


def collision_check(p1, p2, obstacles, safe_dist=0.15):
    for ox, oy, oz, r in obstacles:
        for z in np.linspace(0, 2.5, 15):
            if line_sphere_collision(p1, p2, (ox, oy, z), r + safe_dist):
                return False
    return True


def line_sphere_collision(p1, p2, center, radius):
    p1 = np.array(p1)
    p2 = np.array(p2)
    center = np.array(center)
    d = p2 - p1
    f = p1 - center
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - radius ** 2
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return False
    discriminant = math.sqrt(discriminant)
    t1 = (-b - discriminant) / (2 * a)
    t2 = (-b + discriminant) / (2 * a)
    return 0 <= t1 <= 1 or 0 <= t2 <= 1


def rrt_star(start, goal, obstacles, x_range, y_range, z_range, max_iter=800, step_size=0.3, goal_sample_rate=0.3,
             radius=2.0):
    start_node = Node(start)
    goal_node = Node(goal)
    nodes = [start_node]

    for _ in range(max_iter):
        rnd_point = goal if random.random() < goal_sample_rate else (
            random.uniform(*x_range),
            random.uniform(*y_range),
            random.uniform(*z_range)
        )

        if rnd_point[2] > 2.4:
            rnd_point = (rnd_point[0], rnd_point[1], 2.4)

        nearest_node = get_nearest(nodes, rnd_point)
        new_point = steer(nearest_node.point, rnd_point, step_size)

        if not collision_check(nearest_node.point, new_point, obstacles):
            continue

        new_node = Node(new_point)
        new_node.parent = nearest_node
        new_node.cost = nearest_node.cost + distance(nearest_node.point, new_node.point)

        nearby = get_nearby_nodes(nodes, new_node, radius)
        best_parent = nearest_node
        min_cost = new_node.cost

        for near_node in nearby:
            if collision_check(near_node.point, new_node.point, obstacles):
                cost = near_node.cost + distance(near_node.point, new_node.point)
                if cost < min_cost:
                    best_parent = near_node
                    min_cost = cost

        new_node.parent = best_parent
        new_node.cost = min_cost
        nodes.append(new_node)

        rewire(nodes, new_node, nearby)

        if distance(new_node.point, goal_node.point) < step_size:
            goal_node.parent = new_node
            goal_node.cost = new_node.cost + distance(new_node.point, goal_node.point)
            nodes.append(goal_node)
            break

    path = []
    node = goal_node
    while node.parent is not None:
        path.append(node.point)
        node = node.parent
    path.append(start_node.point)
    path.reverse()

    return path, nodes


def suavizar_caminho(caminho, obstaculos, num_iteracoes=150):
    if len(caminho) <= 2:
        return caminho

    caminho_suave = caminho.copy()

    for _ in range(num_iteracoes):
        i = random.randint(1, len(caminho_suave) - 2)

        ponto_anterior = caminho_suave[i - 1]
        ponto_atual = caminho_suave[i]
        ponto_posterior = caminho_suave[i + 1]

        peso = 0.5
        ponto_novo = tuple((1 - peso) * np.array(ponto_atual) +
                           peso * (np.array(ponto_anterior) + np.array(ponto_posterior)) / 2)

        if collision_check(ponto_anterior, ponto_novo, obstaculos) and \
                collision_check(ponto_novo, ponto_posterior, obstaculos):
            caminho_suave[i] = ponto_novo

    return caminho_suave


# ---------- PARTE 7: INTEGRADOR RRT ------------
class IntegradorRRT:
    def __init__(self, integrador_visao):
        self.integrador_visao = integrador_visao
        self.caminho_atual = []
        self.waypoints = []
        self.lados_passagem = []

    def atualizar_planejamento(self, posicao_atual, destino, ordem_cores, lado_inicial):
        """
        Atualiza o planejamento de rota com base nas hastes detectadas

        Parâmetros:
        - posicao_atual: posição atual do drone (x, y, z)
        - destino: posição de destino (x, y, z)
        - ordem_cores: lista com a ordem das cores das hastes
        - lado_inicial: lado inicial para passar pela primeira haste

        Retorna:
        - caminho atualizado
        """
        # Atualizar posições das hastes
        self.integrador_visao.atualizar_posicoes(posicao_atual)

        # Classificar cores das hastes
        self.integrador_visao.classificar_cores()

        # Obter traves coloridas
        traves = self.integrador_visao.obter_traves_coloridas()

        # Verificar se temos hastes suficientes
        if len(traves) < len(ordem_cores):
            print(f"Aviso: Apenas {len(traves)} hastes detectadas, esperava {len(ordem_cores)}")

        # Gerar waypoints para a missão
        self.waypoints, self.lados_passagem = self.gerar_waypoints_slalom(
            traves,
            ordem_cores,
            lado_inicial,
            deslocamento=1.0,
            altura=1.5
        )

        # Obter obstáculos
        obstaculos = self.integrador_visao.obter_obstaculos()

        # Calcular caminho usando RRT*
        self.caminho_atual = self.calcular_caminho_rrt(
            posicao_atual,
            self.waypoints,
            obstaculos
        )

        return self.caminho_atual

    def gerar_waypoints_slalom(self, traves, ordem_cores, lado_inicial, deslocamento=1.0, altura=1.5):
        """
        Gera waypoints para a missão slalom
        """
        traves_ordenadas = []
        for cor in ordem_cores:
            for trave in traves:
                if trave["cor"] == cor:
                    traves_ordenadas.append(trave)
                    break

        waypoints = []
        lados_passagem = []
        lado = lado_inicial

        # Posição atual como ponto de partida
        last_point = (traves_ordenadas[0]["x"] - 2, traves_ordenadas[0]["y"]) if traves_ordenadas else (0, 0)
        waypoints.append((last_point[0], last_point[1], 0))
        waypoints.append((last_point[0], last_point[1], altura))

        for trave in traves_ordenadas:
            # Vetor direção do movimento (trave - last_point)
            dir_x = trave["x"] - last_point[0]
            dir_y = trave["y"] - last_point[1]
            mag = math.sqrt(dir_x ** 2 + dir_y ** 2)
            if mag > 0:
                dir_x /= mag
                dir_y /= mag
            else:
                dir_x, dir_y = 1, 0  # Direção padrão se não houver movimento

            # Perpendicular (90°): sentido horário ou anti-horário
            if lado == "direito":
                perp_x = dir_y
                perp_y = -dir_x
            else:
                perp_x = -dir_y
                perp_y = dir_x

            # Aplicar deslocamento na perpendicular
            deslocado_x = trave["x"] + deslocamento * perp_x
            deslocado_y = trave["y"] + deslocamento * perp_y
            waypoint = (deslocado_x, deslocado_y, altura)

            waypoints.append(waypoint)
            lados_passagem.append(lado.upper())

            print(f"Passando pela trave {trave['cor'].upper()} pelo lado {lado.upper()}")

            last_point = (trave["x"], trave["y"])  # Atualiza o ponto anterior
            lado = "esquerdo" if lado == "direito" else "direito"

        # Adicionar ponto de pouso
        if traves_ordenadas:
            pouso_x, pouso_y = traves_ordenadas[-1]["x"] + 3, traves_ordenadas[-1]["y"]
        else:
            pouso_x, pouso_y = last_point[0] + 3, last_point[1]

        waypoints.append((pouso_x, pouso_y, altura))
        waypoints.append((pouso_x, pouso_y, 0))

        return waypoints, lados_passagem

    def calcular_caminho_rrt(self, posicao_atual, waypoints, obstaculos):
        """
        Calcula o caminho usando RRT* entre cada par de waypoints
        """
        path_total = []

        # Adicionar posição atual como ponto inicial
        ponto_atual = posicao_atual

        # Calcular caminho para cada waypoint
        for i in range(len(waypoints)):
            destino = waypoints[i]
            print(f"Calculando caminho do ponto {ponto_atual} para {destino}...")

            # Usar RRT* para calcular o caminho
            path, _ = rrt_star(
                ponto_atual, destino, obstaculos,
                x_range=(0, 25), y_range=(0, 10), z_range=(0, 2.4),
                max_iter=800, step_size=0.3, goal_sample_rate=0.3
            )

            # Suavizar o caminho
            path_suave = suavizar_caminho(path, obstaculos, num_iteracoes=150)

            # Adicionar ao caminho total (evitando duplicação de pontos)
            if len(path_total) > 0 and path_suave[0] == path_total[-1]:
                path_total.extend(path_suave[1:])
            else:
                path_total.extend(path_suave)

            # Atualizar ponto atual
            ponto_atual = destino

        return path_total


# ---------- PARTE 8: CONTROLADOR DO DRONE ------------
class ControladorDrone:
    def __init__(self):
        self.posicao = np.array([2.0, 5.0, 0.0])  # [x, y, z] em metros
        self.orientacao = np.array([0.0, 0.0, 0.0])  # [roll, pitch, yaw] em radianos
        self.velocidade = np.array([0.0, 0.0, 0.0])  # [vx, vy, vz] em m/s

        self.caminho_atual = []
        self.indice_waypoint = 0
        self.em_movimento = False
        self.thread_controle = None
        self.running = False

        # Parâmetros de controle
        self.velocidade_max = 1.0  # m/s
        self.distancia_chegada = 0.2  # metros
        self.taxa_atualizacao = 0.1  # segundos

    def iniciar(self):
        """Inicia o controlador do drone"""
        self.running = True
        self.thread_controle = threading.Thread(target=self._loop_controle)
        self.thread_controle.daemon = True
        self.thread_controle.start()
        return True

    def parar(self):
        """Para o controlador do drone"""
        self.running = False
        if self.thread_controle:
            self.thread_controle.join()

    def definir_caminho(self, caminho):
        """Define um novo caminho para o drone seguir"""
        self.caminho_atual = caminho
        self.indice_waypoint = 0
        self.em_movimento = True
        print(f"Novo caminho definido com {len(caminho)} waypoints")

    def _loop_controle(self):
        """Loop principal de controle do drone"""
        while self.running:
            if self.em_movimento and self.caminho_atual and self.indice_waypoint < len(self.caminho_atual):
                # Obter próximo waypoint
                waypoint = self.caminho_atual[self.indice_waypoint]

                # Calcular vetor direção
                direcao = np.array(waypoint) - self.posicao
                distancia = np.linalg.norm(direcao)

                if distancia < self.distancia_chegada:
                    # Chegamos ao waypoint atual
                    print(f"Chegou ao waypoint {self.indice_waypoint}: {waypoint}")
                    self.indice_waypoint += 1

                    if self.indice_waypoint >= len(self.caminho_atual):
                        print("Caminho completo!")
                        self.em_movimento = False
                else:
                    # Normalizar direção e aplicar velocidade
                    if distancia > 0:
                        direcao = direcao / distancia

                    # Calcular velocidade (proporcional à distância, com limite)
                    velocidade_desejada = min(distancia, self.velocidade_max)
                    self.velocidade = direcao * velocidade_desejada

                    # Atualizar posição (simulação simples)
                    self.posicao = self.posicao + self.velocidade * self.taxa_atualizacao

                    # Calcular orientação (yaw) com base na direção do movimento
                    if np.linalg.norm(self.velocidade[:2]) > 0.1:
                        self.orientacao[2] = np.arctan2(self.velocidade[1], self.velocidade[0])

            time.sleep(self.taxa_atualizacao)

    def obter_posicao(self):
        """Retorna a posição atual do drone"""
        return self.posicao.copy()

    def obter_orientacao(self):
        """Retorna a orientação atual do drone"""
        return self.orientacao.copy()

    def obter_status(self):
        """Retorna o status atual do drone"""
        return {
            "posicao": self.posicao.copy(),
            "orientacao": self.orientacao.copy(),
            "velocidade": self.velocidade.copy(),
            "em_movimento": self.em_movimento,
            "waypoint_atual": self.indice_waypoint if self.em_movimento else -1,
            "total_waypoints": len(self.caminho_atual)
        }


# ---------- PARTE 9: VISUALIZAÇÃO 3D ------------
def segmentar_caminho_por_lado(path, traves, lados_passagem, ordem_cores):
    """
    Divide o caminho em segmentos baseados no lado de passagem de cada trave
    """
    # Ordenar as traves conforme a sequência de cores fornecida
    traves_ordenadas = []
    for cor in ordem_cores:
        for trave in traves:
            if trave["cor"] == cor:
                traves_ordenadas.append(trave)
                break

    # Inicializar segmentos
    segmentos_direito = []
    segmentos_esquerdo = []

    # Encontrar os pontos do caminho mais próximos de cada trave
    pontos_trave = []
    for trave in traves_ordenadas:
        # Encontrar o ponto do caminho mais próximo da trave
        min_dist = float('inf')
        idx_mais_proximo = 0
        for i, ponto in enumerate(path):
            dist = np.sqrt((ponto[0] - trave["x"]) ** 2 + (ponto[1] - trave["y"]) ** 2)
            if dist < min_dist:
                min_dist = dist
                idx_mais_proximo = i
        pontos_trave.append(idx_mais_proximo)

    # Adicionar início e fim do caminho
    pontos_trave = [0] + pontos_trave + [len(path) - 1]

    # Criar segmentos baseados nos lados de passagem
    for i in range(len(traves_ordenadas)):
        inicio = pontos_trave[i]
        fim = pontos_trave[i + 2]  # +2 porque pontos_trave inclui o início e o fim

        # Determinar o lado de passagem
        lado = lados_passagem[i]

        # Adicionar ao segmento apropriado
        if lado == "DIREITO":
            segmentos_direito.append((inicio, fim))
        else:
            segmentos_esquerdo.append((inicio, fim))

    return segmentos_direito, segmentos_esquerdo


def draw_3d(traves, raio, path_total, waypoints=None, lados_passagem=None, base_decolagem=None, ordem_cores=None):
    """
    Visualização 3D melhorada com waypoints destacados e caminho segmentado por lado de passagem
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Desenhar as traves
    for idx, trave in enumerate(traves):
        cor_plot = {
            "vermelho": "red",
            "azul": "#135299",
            "rosa": "#f0b19b",
            "preto": "#1a1a1a"
        }.get(trave["cor"], "gray")

        # Desenhar o cilindro da trave
        z = np.linspace(0, 2.5, 20)
        theta = np.linspace(0, 2 * np.pi, 20)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = raio * np.cos(theta_grid) + trave["x"]
        y_grid = raio * np.sin(theta_grid) + trave["y"]
        ax.plot_surface(x_grid, y_grid, z_grid, color=cor_plot, alpha=0.7)

        # Adicionar texto com a cor da trave
        ax.scatter(trave["x"], trave["y"], 2.5, color=cor_plot, s=50)
        ax.text(trave["x"], trave["y"], 2.7, trave["cor"], fontsize=9, ha='center', weight='bold')

        # Adicionar informação sobre o lado de passagem
        if lados_passagem and idx < len(lados_passagem):
            lado = lados_passagem[idx]
            dx = 1 if lado == "DIREITO" else -1
            ax.text(trave["x"] + dx, trave["y"], 1.7, f"Lado: {lado}",
                    fontsize=9, color='black', ha='center', weight='bold',
                    bbox=dict(facecolor='white', alpha=0.7))

    # Desenhar a base de decolagem (círculo azul)
    if base_decolagem:
        x, y = base_decolagem
        # Desenhar quadrado da base
        square_size = 1.0
        x_square = [x - square_size / 2, x + square_size / 2, x + square_size / 2, x - square_size / 2,
                    x - square_size / 2]
        y_square = [y - square_size / 2, y - square_size / 2, y + square_size / 2, y + square_size / 2,
                    y - square_size / 2]
        z_square = [0, 0, 0, 0, 0]
        ax.plot(x_square, y_square, z_square, 'k-')

        # Desenhar círculo azul
        theta = np.linspace(0, 2 * np.pi, 30)
        radius = np.linspace(0, 0.4, 2)  # Raio do círculo
        theta_grid, r_grid = np.meshgrid(theta, radius)

        x_circle = r_grid * np.cos(theta_grid) + x
        y_circle = r_grid * np.sin(theta_grid) + y
        z_circle = np.zeros_like(x_circle)

        ax.plot_surface(x_circle, y_circle, z_circle, color='blue', alpha=0.5)
        ax.text(x, y, 0.1, "Base", fontsize=9, ha='center', color='white')

    # Segmentar o caminho por lado de passagem
    if path_total and lados_passagem and ordem_cores:
        segmentos_direito, segmentos_esquerdo = segmentar_caminho_por_lado(
            path_total, traves, lados_passagem, ordem_cores
        )

        # Desenhar primeiro os segmentos do lado esquerdo (atrás das traves)
        for inicio, fim in segmentos_esquerdo:
            segmento = path_total[inicio:fim + 1]
            x, y, z = zip(*segmento)
            ax.plot(x, y, z, '-', color='green', linewidth=2, alpha=0.7)

        # Desenhar as traves novamente para garantir que fiquem na frente dos segmentos esquerdos
        for trave in traves:
            cor_plot = {
                "vermelho": "red",
                "azul": "#135299",
                "rosa": "#f0b19b",
                "preto": "#1a1a1a"
            }.get(trave["cor"], "gray")

            z = np.linspace(0, 2.5, 20)
            theta = np.linspace(0, 2 * np.pi, 20)
            theta_grid, z_grid = np.meshgrid(theta, z)
            x_grid = raio * np.cos(theta_grid) + trave["x"]
            y_grid = raio * np.sin(theta_grid) + trave["y"]
            ax.plot_surface(x_grid, y_grid, z_grid, color=cor_plot, alpha=0.7)

        # Desenhar depois os segmentos do lado direito (na frente das traves)
        for inicio, fim in segmentos_direito:
            segmento = path_total[inicio:fim + 1]
            x, y, z = zip(*segmento)
            ax.plot(x, y, z, '-', color='green', linewidth=2, alpha=1.0)
    else:
        # Desenhar o caminho completo se não for possível segmentar
        x, y, z = zip(*path_total)
        ax.plot(x, y, z, '-', color='green', linewidth=2, label="Trajeto Suavizado")

    # Destacar pontos de decolagem e pouso
    ax.scatter(path_total[0][0], path_total[0][1], path_total[0][2], color='blue', s=100, marker='^', label="Decolagem")
    ax.scatter(path_total[-1][0], path_total[-1][1], path_total[-1][2], color='red', s=100, marker='v', label="Pouso")

    # Destacar os waypoints
    if waypoints:
        waypoints_x, waypoints_y, waypoints_z = zip(*waypoints)
        ax.scatter(waypoints_x, waypoints_y, waypoints_z, color='orange', s=80, marker='*', label="Waypoints")

    # Adicionar plano do solo
    x_min, x_max = 0, 25
    y_min, y_max = 0, 10
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, color='lightgray', alpha=0.3)

    # Adicionar linha de altura máxima (2.5m)
    x_grid = np.linspace(x_min, x_max, 2)
    y_grid = np.linspace(y_min, y_max, 2)
    x_plane, y_plane = np.meshgrid(x_grid, y_grid)
    z_plane = np.ones_like(x_plane) * 2.5
    ax.plot_surface(x_plane, y_plane, z_plane, color='red', alpha=0.1)
    ax.text(x_min, y_min, 2.5, "Altura Máxima (2.5m)", color='red', fontsize=8)

    # Adicionar linha de altura de voo (1.5m)
    z_plane_flight = np.ones_like(x_plane) * 1.5
    ax.plot_surface(x_plane, y_plane, z_plane_flight, color='blue', alpha=0.1)
    ax.text(x_min, y_min, 1.5, "Altura de Voo (1.5m)", color='blue', fontsize=8)

    # Configurar limites e rótulos
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(0, 3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Missão Slalom - Navegação entre Traves Coloridas")

    # Adicionar legenda
    plt.legend(loc='upper right')

    # Adicionar informações da missão como texto na figura
    info_text = (
        "Missão: Navegação entre traves coloridas\n"
        f"Sequência: {', '.join(ordem_cores)}\n"
        f"Lado inicial: {lados_passagem[0] if lados_passagem else 'N/A'}\n"
        "Requisitos: Altitude < 2.5m, alternância de lados\n"
        f"Diâmetro das traves: {raio * 2 * 100:.0f}cm"
    )
    plt.figtext(0.02, 0.02, info_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

    # Adicionar legenda para os lados de passagem
    legenda_lados = (
        "Visualização dos lados de passagem:\n"
        "- Lado DIREITO: Caminho na frente da trave\n"
        "- Lado ESQUERDO: Caminho atrás da trave"
    )
    plt.figtext(0.02, 0.15, legenda_lados, fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.show()


# ---------- PARTE 10: FUNÇÃO PRINCIPAL ------------
def main():
    print("Iniciando sistema integrado de visão computacional e controle de voo...")

    # Modo de operação: 'camera', 'simulacao' ou 'teste_camera'
    modo = input("Escolha o modo de operação (camera/simulacao/teste_camera): ").lower()

    # Inicializar detector de hastes (apenas no modo câmera ou teste_camera)
    detector = None
    if modo in ['camera', 'teste_camera']:
        detector = DetectorHastes(camera_id=0)
        if not detector.iniciar():
            print("Falha ao iniciar o detector de hastes!")
            return

    # Inicializar conversor de coordenadas
    conversor = ConversorCoordenadas(
        altura_camera=1.5,
        fov_horizontal=60,
        fov_vertical=45,
        resolucao=(640, 480)
    )

    # Inicializar integrador de visão e rota (apenas no modo câmera ou teste_camera)
    integrador_visao = None
    if modo in ['camera', 'teste_camera']:
        integrador_visao = IntegradorVisaoRota(detector, conversor, raio_hastes=0.05)

    # Inicializar controlador do drone
    controlador = ControladorDrone()
    controlador.iniciar()

    # Definir parâmetros da missão
    ordem_cores = ["preto", "azul", "vermelho", "rosa"]
    lado_inicial = "direito"
    destino = (20, 5, 0)  # Destino final

    # Modo de teste com câmera - detecta hastes e gera visualização 3D
    if modo == 'teste_camera':
        print("\nModo de teste com câmera. Pressione 'q' para sair, 'c' para capturar e visualizar")

        while True:
            # Processar comandos do usuário
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                print("\nCapturando hastes e gerando visualização 3D...")

                # Obter posição atual do drone
                posicao_drone = controlador.obter_posicao()

                # Atualizar posições das hastes
                integrador_visao.atualizar_posicoes(posicao_drone)

                # Classificar cores das hastes
                cores = integrador_visao.classificar_cores()

                # Obter traves coloridas
                traves = integrador_visao.obter_traves_coloridas()

                if not traves:
                    print("Nenhuma haste detectada. Tente novamente.")
                    continue

                print(f"Hastes detectadas: {len(traves)}")
                for i, trave in enumerate(traves):
                    print(f"  Haste {i + 1}: Posição ({trave['x']:.2f}, {trave['y']:.2f}), Cor: {trave['cor']}")

                # Inicializar integrador RRT para gerar waypoints
                integrador_rrt = IntegradorRRT(integrador_visao)

                # Gerar waypoints e caminho
                caminho = integrador_rrt.atualizar_planejamento(
                    posicao_drone, destino, ordem_cores, lado_inicial
                )

                # Visualizar a missão em 3D
                draw_3d(
                    traves,
                    integrador_visao.raio_hastes,
                    caminho,
                    integrador_rrt.waypoints,
                    integrador_rrt.lados_passagem,
                    (posicao_drone[0], posicao_drone[1]),  # Base de decolagem
                    ordem_cores
                )

            # Exibir frame com detecções
            frame = detector.obter_frame()
            cv2.imshow("Detecção de Hastes", frame)

            # Pequena pausa para reduzir uso de CPU
            time.sleep(0.05)

    # Parâmetros para simulação
    elif modo == 'simulacao':
        # Definir as traves conforme a missão
        traves = [
            {"x": 6, "y": 4, "cor": "preto"},
            {"x": 10, "y": 5, "cor": "azul"},
            {"x": 14, "y": 4, "cor": "vermelho"},
            {"x": 18, "y": 5, "cor": "rosa"},
        ]

        # Raio das traves (em metros) - 50mm = 100mm de diâmetro
        raio_trave = 0.05

        # Posição da base de decolagem
        base_decolagem = (2, 5)

        # Gerar waypoints para a missão
        waypoints = []
        lados_passagem = []

        # Posição inicial
        last_point = (base_decolagem[0], base_decolagem[1])
        waypoints.append((last_point[0], last_point[1], 0))
        waypoints.append((last_point[0], last_point[1], 1.5))

        lado = lado_inicial
        for trave in traves:
            # Vetor direção do movimento
            dir_x = trave["x"] - last_point[0]
            dir_y = trave["y"] - last_point[1]
            mag = math.sqrt(dir_x ** 2 + dir_y ** 2)
            dir_x /= mag
            dir_y /= mag

            # Perpendicular
            if lado == "direito":
                perp_x = dir_y
                perp_y = -dir_x
            else:
                perp_x = -dir_y
                perp_y = dir_x

            # Aplicar deslocamento
            deslocado_x = trave["x"] + 1.0 * perp_x
            deslocado_y = trave["y"] + 1.0 * perp_y
            waypoint = (deslocado_x, deslocado_y, 1.5)

            waypoints.append(waypoint)
            lados_passagem.append(lado.upper())

            last_point = (trave["x"], trave["y"])
            lado = "esquerdo" if lado == "direito" else "direito"

        # Adicionar ponto de pouso
        pouso_x, pouso_y = traves[-1]["x"] + 3, traves[-1]["y"]
        waypoints.append((pouso_x, pouso_y, 1.5))
        waypoints.append((pouso_x, pouso_y, 0))

        # Definir obstáculos (traves)
        obstaculos = [(t["x"], t["y"], 0, raio_trave) for t in traves]

        # Calcular caminho usando RRT*
        path_total = []
        for i in range(len(waypoints) - 1):
            print(f"Calculando caminho do waypoint {i} para {i + 1}...")
            segment, _ = rrt_star(
                waypoints[i], waypoints[i + 1], obstaculos,
                x_range=(0, 25), y_range=(0, 10), z_range=(0, 2.4),
                max_iter=800, step_size=0.3, goal_sample_rate=0.3
            )

            # Adicionar segmento ao caminho total
            if len(path_total) > 0 and segment[0] == path_total[-1]:
                path_total.extend(segment[1:])
            else:
                path_total.extend(segment)

        # Suavizar o caminho
        path_suavizado = suavizar_caminho(path_total, obstaculos, num_iteracoes=150)

        # Visualizar a missão em 3D
        print("\n=== VISUALIZANDO SIMULAÇÃO 3D ===")
        draw_3d(traves, raio_trave, path_suavizado, waypoints, lados_passagem, base_decolagem, ordem_cores)

        # Definir o caminho no controlador
        controlador.definir_caminho(path_suavizado)

        # Iniciar movimento
        controlador.em_movimento = True

        # Loop principal da simulação
        try:
            while controlador.em_movimento:
                # Atualizar status
                status = controlador.obter_status()
                print(
                    f"Posição: ({status['posicao'][0]:.1f}, {status['posicao'][1]:.1f}, {status['posicao'][2]:.1f}) | " +
                    f"Waypoint: {status['waypoint_atual']}/{status['total_waypoints']}")

                time.sleep(0.5)

            print("\nSimulação concluída com sucesso!")
        except KeyboardInterrupt:
            print("Interrompido pelo usuário")

    # Modo câmera normal
    elif modo == 'camera':
        try:
            # Estado da missão
            estado_missao = "INICIALIZANDO"  # INICIALIZANDO, PLANEJANDO, EXECUTANDO, CONCLUIDO
            ultima_atualizacao = time.time()
            intervalo_atualizacao = 5.0  # segundos

            print(
                "Sistema iniciado. Pressione 'q' para sair, 'p' para planejar, 'e' para executar, 'v' para visualizar")

            # Inicializar integrador RRT
            integrador_rrt = IntegradorRRT(integrador_visao)

            while True:
                # Obter posição atual do drone
                posicao_drone = controlador.obter_posicao()

                # Atualizar estado da missão
                tempo_atual = time.time()
                if tempo_atual - ultima_atualizacao > intervalo_atualizacao:
                    if estado_missao == "EXECUTANDO":
                        # Atualizar detecção de hastes e replanejamento periódico
                        integrador_visao.atualizar_posicoes(posicao_drone)
                        integrador_visao.classificar_cores()

                        # Verificar se é necessário replanejar
                        if controlador.indice_waypoint < len(controlador.caminho_atual) // 2:
                            print("Atualizando planejamento durante o voo...")
                            caminho = integrador_rrt.atualizar_planejamento(
                                posicao_drone, destino, ordem_cores, lado_inicial
                            )

                            # Atualizar caminho apenas se for significativamente diferente
                            if len(caminho) > 0 and len(controlador.caminho_atual) > 0:
                                diferenca = np.mean(np.abs(np.array(caminho) - np.array(controlador.caminho_atual)))
                                if diferenca > 0.5:
                                    print(f"Diferença significativa detectada ({diferenca:.2f}m). Atualizando caminho.")
                                    controlador.definir_caminho(caminho)

                    ultima_atualizacao = tempo_atual

                # Processar comandos do usuário
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p') and estado_missao != "EXECUTANDO":
                    print("\nPlanejando rota...")
                    estado_missao = "PLANEJANDO"

                    # Atualizar planejamento
                    caminho = integrador_rrt.atualizar_planejamento(
                        posicao_drone, destino, ordem_cores, lado_inicial
                    )

                    print(f"Caminho planejado com {len(caminho)} pontos")
                    print("Waypoints:")
                    for i, wp in enumerate(integrador_rrt.waypoints):
                        print(f"  {i}: ({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f})")

                    # Definir o caminho no controlador (mas não iniciar movimento ainda)
                    controlador.definir_caminho(caminho)
                    controlador.em_movimento = False

                    estado_missao = "PLANEJADO"

                elif key == ord('e') and estado_missao == "PLANEJADO":
                    print("\nIniciando execução da missão...")
                    estado_missao = "EXECUTANDO"

                    # Iniciar movimento
                    controlador.em_movimento = True

                elif key == ord('v') and estado_missao == "PLANEJADO":
                    print("\nVisualizando caminho planejado em 3D...")

                    # Obter traves coloridas
                    traves = integrador_visao.obter_traves_coloridas()

                    # Visualizar a missão em 3D
                    draw_3d(
                        traves,
                        integrador_visao.raio_hastes,
                        controlador.caminho_atual,
                        integrador_rrt.waypoints,
                        integrador_rrt.lados_passagem,
                        (2, 5),  # Base de decolagem
                        ordem_cores
                    )

                # Exibir status
                frame = detector.obter_frame()
                status = controlador.obter_status()
                cv2.putText(
                    frame,
                    f"Estado: {estado_missao} | Pos: ({status['posicao'][0]:.1f}, {status['posicao'][1]:.1f}, {status['posicao'][2]:.1f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

                # Verificar se a missão foi concluída
                if estado_missao == "EXECUTANDO" and not controlador.em_movimento:
                    print("\nMissão concluída com sucesso!")
                    estado_missao = "CONCLUIDO"

                # Pequena pausa para reduzir uso de CPU
                time.sleep(0.05)

        except KeyboardInterrupt:
            print("Interrompido pelo usuário")

    # Limpar recursos
    if detector:
        detector.parar()
    controlador.parar()
    cv2.destroyAllWindows()
    print("Sistema finalizado")


if __name__ == "__main__":
    main()