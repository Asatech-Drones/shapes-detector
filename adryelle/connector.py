import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
import numpy as np
import time
import sys
import os
import threading
import cv2

# Importar as classes do umaHasteVoo.py
# Assumindo que o arquivo está no mesmo diretório
from umaHasteVoo import (
    DetectorHastes,
    ConversorCoordenadas,
    IntegradorVisaoRota,
    IntegradorRRT,
    draw_3d,
    rrt_star,
    suavizar_caminho
)


class MAVSDKConnector:
    """
    Classe para conectar o sistema de detecção de hastes ao MAVSDK
    para controle do drone no Gazebo/PX4/QGroundControl
    """

    def __init__(self, mavsdk_server_address="localhost", mavsdk_server_port=50051):
        self.drone = None
        self.mavsdk_server_address = mavsdk_server_address
        self.mavsdk_server_port = mavsdk_server_port

        self.posicao_atual = np.array([0.0, 0.0, 0.0])  # [x, y, z] em metros (NED)
        self.orientacao_atual = np.array([0.0, 0.0, 0.0])  # [roll, pitch, yaw] em radianos
        self.velocidade_atual = np.array([0.0, 0.0, 0.0])  # [vx, vy, vz] em m/s

        self.caminho_atual = []
        self.indice_waypoint = 0
        self.em_movimento = False
        self.running = False
        self.thread_telemetria = None

        # Parâmetros de controle
        self.velocidade_max = 1.0  # m/s
        self.distancia_chegada = 0.2  # metros
        self.taxa_atualizacao = 0.1  # segundos

        # Detector e integrador
        self.detector = None
        self.integrador_visao = None
        self.integrador_rrt = None

        # Lock para acesso seguro aos dados compartilhados
        self.lock = threading.Lock()

    async def conectar(self):
        """Conecta ao drone via MAVSDK"""
        print(f"Conectando ao servidor MAVSDK em {self.mavsdk_server_address}:{self.mavsdk_server_port}...")
        self.drone = System(mavsdk_server_address=self.mavsdk_server_address, port=self.mavsdk_server_port)

        # Não precisamos especificar system_address, pois o mavsdk_server já está conectado
        # ao simulador na porta UDP 18750
        await self.drone.connect()

        print("Aguardando conexão...")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("Drone conectado!")
                break

        # Iniciar thread de telemetria
        self.running = True
        self.thread_telemetria = threading.Thread(target=lambda: asyncio.run(self._loop_telemetria()))
        self.thread_telemetria.daemon = True
        self.thread_telemetria.start()

        return True

    async def _loop_telemetria(self):
        """Loop para atualizar telemetria do drone"""
        while self.running:
            try:
                # Obter posição atual
                async for position in self.drone.telemetry.position():
                    # Converter de NED para coordenadas do mundo
                    with self.lock:
                        self.posicao_atual = np.array([
                            position.north_m,  # x
                            position.east_m,  # y
                            -position.down_m  # z (converter de down para up)
                        ])
                    break

                # Obter orientação atual
                async for attitude in self.drone.telemetry.attitude_euler():
                    with self.lock:
                        self.orientacao_atual = np.array([
                            attitude.roll_deg * np.pi / 180.0,
                            attitude.pitch_deg * np.pi / 180.0,
                            attitude.yaw_deg * np.pi / 180.0
                        ])
                    break

                # Obter velocidade atual
                async for velocity in self.drone.telemetry.velocity_ned():
                    with self.lock:
                        self.velocidade_atual = np.array([
                            velocity.north_m_s,
                            velocity.east_m_s,
                            -velocity.down_m_s
                        ])
                    break

                await asyncio.sleep(0.1)  # Atualizar a 10Hz

            except Exception as e:
                print(f"Erro na telemetria: {e}")
                await asyncio.sleep(1.0)

    async def armar_e_decolar(self, altitude=1.5):
        """Arma o drone e realiza a decolagem"""
        print("Verificando se o drone está pronto...")
        async for health in self.drone.telemetry.health():
            if health.is_armable:
                print("Drone pronto para armar!")
                break
            print("Aguardando drone ficar pronto para armar...")
            await asyncio.sleep(1)

        print("Armando drone...")
        await self.drone.action.arm()

        print(f"Definindo altitude de decolagem para {altitude}m...")
        await self.drone.action.set_takeoff_altitude(altitude)

        print("Decolando...")
        await self.drone.action.takeoff()

        # Aguardar estabilização após decolagem
        print("Aguardando estabilização...")
        await asyncio.sleep(5)

        return True

    async def pousar(self):
        """Inicia o procedimento de pouso do drone"""
        print("Iniciando pouso...")

        # Sair do modo offboard antes de pousar
        try:
            await self.drone.offboard.stop()
            print("Modo Offboard encerrado.")
        except:
            print("Erro ao encerrar modo Offboard, continuando com pouso...")

        await self.drone.action.land()
        print("Comando de pouso enviado.")

        # Aguardar o pouso
        print("Aguardando pouso completo...")
        async for is_flying in self.drone.telemetry.in_air():
            if not is_flying:
                print("Pouso completo!")
                break
            await asyncio.sleep(1)

        return True

    async def iniciar_offboard(self):
        """Inicia o modo Offboard para controle de velocidade"""
        print("Iniciando modo Offboard...")

        # Definir velocidade inicial como zero
        await self.drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
        )

        # Tentar iniciar o modo Offboard
        try:
            await self.drone.offboard.start()
            print("Modo Offboard iniciado com sucesso.")
        except OffboardError as error:
            print(f"Erro ao iniciar modo Offboard: {error}")
            print("Tentando novamente após definir velocidade...")
            await self.drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
            )
            await asyncio.sleep(0.5)
            await self.drone.offboard.start()
            print("Modo Offboard iniciado com sucesso na segunda tentativa.")

        return True

    async def seguir_caminho(self, caminho, velocidade_max=0.5, distancia_chegada=0.2):
        """Segue um caminho de waypoints usando o controlador MAVSDK"""
        if not caminho:
            print("Caminho vazio, nada a fazer.")
            return False

        print(f"Iniciando navegação com {len(caminho)} pontos...")

        # Iniciar modo Offboard
        await self.iniciar_offboard()

        # Seguir cada waypoint
        for i, waypoint in enumerate(caminho):
            print(f"Indo para waypoint {i + 1}/{len(caminho)}: {waypoint}")

            chegou = False
            while not chegou and self.running:
                # Obter posição atual
                with self.lock:
                    posicao_atual = self.posicao_atual.copy()

                # Calcular vetor direção
                waypoint_np = np.array(waypoint)
                direcao = waypoint_np - posicao_atual
                distancia = np.linalg.norm(direcao)

                if distancia < distancia_chegada:
                    print(f"Chegou ao waypoint {i + 1}")
                    chegou = True
                    # Parar brevemente ao atingir o waypoint
                    await self.drone.offboard.set_velocity_body(
                        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
                    )
                    await asyncio.sleep(0.5)
                else:
                    # Normalizar direção
                    if distancia > 0:
                        direcao = direcao / distancia

                    # Calcular velocidade (proporcional à distância, com limite)
                    velocidade_desejada = min(distancia, velocidade_max)
                    velocidade = direcao * velocidade_desejada

                    # Converter para velocidade no corpo do drone
                    # Precisamos converter do sistema NED para o sistema do corpo do drone
                    # Isso é uma simplificação, para maior precisão precisaríamos usar quaternions
                    with self.lock:
                        yaw = self.orientacao_atual[2]  # em radianos

                    # Matriz de rotação para converter de NED para corpo do drone
                    cos_yaw = np.cos(yaw)
                    sin_yaw = np.sin(yaw)

                    # Velocidade no corpo do drone
                    vx = velocidade[0] * cos_yaw + velocidade[1] * sin_yaw
                    vy = -velocidade[0] * sin_yaw + velocidade[1] * cos_yaw
                    vz = -velocidade[2]  # Converter de NED (down) para corpo (up)

                    # Enviar comando de velocidade
                    await self.drone.offboard.set_velocity_body(
                        VelocityBodyYawspeed(vx, vy, vz, 0.0)
                    )

                await asyncio.sleep(0.1)

        print("Caminho completo!")
        await self.drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
        )

        return True

    def definir_caminho(self, caminho):
        """Define um novo caminho para o drone seguir"""
        with self.lock:
            self.caminho_atual = caminho
            self.indice_waypoint = 0
        print(f"Novo caminho definido com {len(caminho)} waypoints")

    def obter_posicao(self):
        """Retorna a posição atual do drone"""
        with self.lock:
            return self.posicao_atual.copy()

    def obter_orientacao(self):
        """Retorna a orientação atual do drone"""
        with self.lock:
            return self.orientacao_atual.copy()

    def obter_status(self):
        """Retorna o status atual do drone"""
        with self.lock:
            return {
                "posicao": self.posicao_atual.copy(),
                "orientacao": self.orientacao_atual.copy(),
                "velocidade": self.velocidade_atual.copy(),
                "em_movimento": self.em_movimento,
                "waypoint_atual": self.indice_waypoint,
                "total_waypoints": len(self.caminho_atual)
            }

    async def finalizar(self):
        """Finaliza a conexão com o drone"""
        self.running = False
        if hasattr(self, 'thread_telemetria') and self.thread_telemetria:
            self.thread_telemetria.join(timeout=2.0)

        if self.drone:
            # Garantir que o drone está em modo seguro
            try:
                await self.drone.offboard.stop()
            except:
                pass

            # Desconectar
            await self.drone.close()
            print("Conexão com o drone finalizada.")


# Função principal para executar a missão
async def executar_missao(mavsdk_server_address="localhost", mavsdk_server_port=50051):
    """Função principal para executar a missão com MAVSDK"""
    print("Iniciando sistema integrado com MAVSDK...")

    # Modo de operação: 'camera', 'simulacao' ou 'teste_camera'
    modo = input("Escolha o modo de operação (camera/simulacao/teste_camera): ").lower()

    # Inicializar conector MAVSDK
    conector = MAVSDKConnector(
        mavsdk_server_address=mavsdk_server_address,
        mavsdk_server_port=mavsdk_server_port
    )

    # Conectar ao drone
    await conector.conectar()

    # Inicializar detector de hastes (apenas no modo câmera ou teste_camera)
    detector = None
    if modo in ['camera', 'teste_camera']:
        detector = DetectorHastes(camera_id=0)
        if not detector.iniciar():
            print("Falha ao iniciar o detector de hastes!")
            await conector.finalizar()
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
        conector.integrador_visao = integrador_visao

    # Definir parâmetros da missão
    ordem_cores = ["preto", "azul", "vermelho", "rosa"]
    lado_inicial = "direito"
    destino = (20, 5, 0)  # Destino final

    try:
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
                    posicao_drone = conector.obter_posicao()

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
                    conector.integrador_rrt = integrador_rrt

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
                if detector:
                    frame = detector.obter_frame()
                    cv2.imshow("Detecção de Hastes", frame)

                # Pequena pausa para reduzir uso de CPU
                await asyncio.sleep(0.05)

        # Modo simulação - usa hastes pré-definidas
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
                mag = np.sqrt(dir_x ** 2 + dir_y ** 2)
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

            # Perguntar se deseja executar a missão
            executar = input("\nDeseja executar a missão no drone? (s/n): ").lower()
            if executar == 's':
                print("\n=== INICIANDO MISSÃO ===")

                # Armar e decolar
                await conector.armar_e_decolar(altitude=1.5)

                # Seguir o caminho
                await conector.seguir_caminho(path_suavizado, velocidade_max=0.5, distancia_chegada=0.2)

                # Pousar
                await conector.pousar()

                print("\n=== MISSÃO CONCLUÍDA ===")

        # Modo câmera normal - detecta hastes e controla o drone
        elif modo == 'camera':
            # Inicializar integrador RRT
            integrador_rrt = IntegradorRRT(integrador_visao)
            conector.integrador_rrt = integrador_rrt

            print("\n=== MODO CÂMERA ATIVO ===")
            print("Pressione 'q' para sair, 'p' para planejar, 'e' para executar, 'v' para visualizar")

            # Estado da missão
            estado_missao = "INICIALIZANDO"  # INICIALIZANDO, PLANEJANDO, EXECUTANDO, CONCLUIDO

            while True:
                # Processar comandos do usuário
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p') and estado_missao != "EXECUTANDO":
                    print("\nPlanejando rota...")
                    estado_missao = "PLANEJANDO"

                    # Obter posição atual do drone
                    posicao_drone = conector.obter_posicao()

                    # Atualizar planejamento
                    caminho = integrador_rrt.atualizar_planejamento(
                        posicao_drone, destino, ordem_cores, lado_inicial
                    )

                    print(f"Caminho planejado com {len(caminho)} pontos")
                    conector.definir_caminho(caminho)

                    estado_missao = "PLANEJADO"

                elif key == ord('e') and estado_missao == "PLANEJADO":
                    print("\nIniciando execução da missão...")
                    estado_missao = "EXECUTANDO"

                    # Armar e decolar
                    await conector.armar_e_decolar(altitude=1.5)

                    # Seguir o caminho
                    await conector.seguir_caminho(conector.caminho_atual)

                    # Pousar
                    await conector.pousar()

                    print("\nMissão concluída!")
                    estado_missao = "CONCLUIDO"

                elif key == ord('v') and estado_missao == "PLANEJADO":
                    print("\nVisualizando caminho planejado em 3D...")

                    # Obter traves coloridas
                    traves = integrador_visao.obter_traves_coloridas()

                    # Visualizar a missão em 3D
                    draw_3d(
                        traves,
                        integrador_visao.raio_hastes,
                        conector.caminho_atual,
                        integrador_rrt.waypoints,
                        integrador_rrt.lados_passagem,
                        (conector.posicao_atual[0], conector.posicao_atual[1]),
                        ordem_cores
                    )

                # Exibir frame com detecções
                if detector:
                    frame = detector.obter_frame()
                    status = conector.obter_status()
                    cv2.putText(
                        frame,
                        f"Estado: {estado_missao} | Pos: ({status['posicao'][0]:.1f}, {status['posicao'][1]:.1f}, {status['posicao'][2]:.1f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )
                    cv2.imshow("Detecção de Hastes", frame)

                # Pequena pausa para reduzir uso de CPU
                await asyncio.sleep(0.05)

    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário")
    except Exception as e:
        print(f"\nErro: {e}")
    finally:
        # Limpar recursos
        if detector:
            detector.parar()

        # Finalizar conexão com o drone
        await conector.finalizar()

        cv2.destroyAllWindows()
        print("Sistema finalizado")


# Função para executar o script diretamente
if __name__ == "__main__":
    # Verificar argumentos de linha de comando para servidor MAVSDK
    mavsdk_server_address = "localhost"
    mavsdk_server_port = 50051

    if len(sys.argv) > 1:
        mavsdk_server_address = sys.argv[1]
    if len(sys.argv) > 2:
        mavsdk_server_port = int(sys.argv[2])

    # Executar a missão
    asyncio.run(executar_missao(mavsdk_server_address, mavsdk_server_port))