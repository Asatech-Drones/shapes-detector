import asyncio
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed
import time

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2
import time

from shape_detector import cap, detect_shape, destroy_windows

# Definições de tolerância para centralização no frame
TOLERANCE = 20
STEP_SIZE = 0.00001
K_ALTITUDE = 0.5

# Inicializar o drone
drone = System()
last_move_time = 0.0
initial_altitude = 0.0

async def teste_movimento(drone):
    print("Movendo para frente...")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.5, 0.0, 0.0, 0.0))  # frente
    await asyncio.sleep(3)

    print("Movendo para trás...")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(-0.5, 0.0, 0.0, 0.0))  # trás
    await asyncio.sleep(3)

    print("Movendo para direita...")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.5, 0.0, 0.0))  # direita
    await asyncio.sleep(3)

    print("Movendo para esquerda...")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, -0.5, 0.0, 0.0))  # esquerda
    await asyncio.sleep(3)

    print("Parando...")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))  # parar


async def connect_drone():
    """ Conecta ao drone e realiza a decolagem. """
    await drone.connect(system_address="udp://:14540")

    print("Esperando conexão com o drone...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone conectado!")
            break
    print("Armando drone...")
    await drone.action.arm()
    print("Decolando...")
    await drone.action.takeoff()
    print("Decolagem concluída...")

    # Esperar o drone estabilizar após a decolagem
    await asyncio.sleep(10)

    global initial_altitude
    async for position in drone.telemetry.position():
        initial_altitude = position.relative_altitude_m
        print(f"Altitude inicial registrada: {initial_altitude:.2f} m")
        break

    # Enviar primeiro comando nulo e iniciar o modo offboard
    print("Iniciando modo Offboard...")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    await drone.offboard.start()
    print("Modo Offboard iniciado com sucesso.")

    await teste_movimento(drone)

    global last_move_time
    last_move_time = time.time()

# PID Configs
Kp = 0.002
Ki = 0.000
Kd = 0.001

MAX_SPEED = 0.5

# Estado do PID para cada eixo
pid_state = {
    "x": {"prev_error": 0, "integral": 0},
    "y": {"prev_error": 0, "integral": 0}
}

async def move_to_target(cx, cy, frame_width, frame_height):
    """ Move o drone para centralizar a figura no frame da câmera usando PID. """

    # Erro entre o centro do frame e o centro da figura detectada
    x_error = frame_width // 2 - cx
    y_error = frame_height // 2 - cy

    # Normaliza (opcional: você pode dividir pelos tamanhos do frame 
    # para deixar o erro proporcional ao tamanho da imagem)
    x_error = int(x_error)
    y_error = int(y_error)

    # Atualiza PID eixo X (direita/esquerda)
    pid_state["x"]["integral"] += x_error
    derivative_x = x_error - pid_state["x"]["prev_error"]
    right_speed = (
        Kp * x_error + Ki * pid_state["x"]["integral"] + Kd * derivative_x
    )
    pid_state["x"]["prev_error"] = x_error

    # Atualiza PID eixo Y (frente/trás)
    pid_state["y"]["integral"] += y_error
    derivative_y = y_error - pid_state["y"]["prev_error"]
    forward_speed = (
        Kp * y_error + Ki * pid_state["y"]["integral"] + Kd * derivative_y
    )
    pid_state["y"]["prev_error"] = y_error

    # Inverte sinais conforme eixo de referência do drone
    right = -right_speed
    forward = +forward_speed

    # Limita a velocidade para segurança
    right = max(min(right, MAX_SPEED), -MAX_SPEED)
    forward = max(min(forward, MAX_SPEED), -MAX_SPEED)

    # Calcula diferença da altitude atual para a inicial
    global initial_altitude
    async for position in drone.telemetry.position():
        altitude_error = initial_altitude - position.relative_altitude_m
        break

    # Calcula a compensação de altitude em relação a altitude inicial
    altitude_compensation = - (K_ALTITUDE * altitude_error)

    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(forward, right, altitude_compensation, 0.0)
    )

    # Verifica se está dentro da tolerância para centralizar
    return abs(x_error) < TOLERANCE and abs(y_error) < TOLERANCE


async def land_drone():
    """ Inicia o procedimento de pouso do drone. """
    print("Centralizado! Iniciando pouso...")
    await drone.action.land()

async def main():
    # """ Fluxo principal do sistema de navegação autônoma. """
    await connect_drone()

    while True:

        success, frame = cap.read()
        if not success:
            break
        xRect, yRect, wRect, hRect = detect_shape(frame)

        centralizado = False
        if(xRect != 0 and yRect != 0):
            cx = xRect + wRect // 2
            cy = yRect + hRect // 2

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Ajusta posição do drone até a figura estar centralizada
            centralizado = await move_to_target(cx, cy, frame_width, frame_height)

        if centralizado:
            await land_drone()
            break
        
        # AQUI: Checa a altitude logo após o movimento
        async for position in drone.telemetry.position():
            print(f"Altitude atual: {position.relative_altitude_m:.2f} m")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Executa a lógica assíncrona
asyncio.run(main())
destroy_windows()