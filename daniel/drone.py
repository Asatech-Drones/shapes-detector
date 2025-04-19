import asyncio
from mavsdk import System

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2
import time

from shape_detector import cap, detect_shape, destroy_windows

# Definições de tolerância para centralização no frame
TOLERANCE = 20  # Pixels
STEP_SIZE = 0.00001  # Pequeno ajuste na latitude/longitude

# Inicializar o drone
drone = System()
last_move_time = 0.0

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

    # Tempo para estabilizar
    await asyncio.sleep(15)

    global last_move_time
    last_move_time = time.time()

# async def detect_figure():
#     """ Simula a detecção da figura no frame da câmera. Retorna (cx, cy). """
#     # Resolução da câmera (simulada)
#     frame_width, frame_height = 640, 480
    
#     # Simula uma detecção aleatória da figura geométrica
#     cx = random.randint(0, frame_width)
#     cy = random.randint(0, frame_height)

#     return cx, cy, frame_width, frame_height

async def move_to_target(cx, cy, frame_width, frame_height):
    """ Move o drone para centralizar a figura no frame da câmera. """
    x_offset = frame_width // 2 - cx
    y_offset = frame_height // 2 - cy

    print(f"offsets: {x_offset}, {x_offset}")
    
    # Obtém posição atual do drone
    async for position in drone.telemetry.position():
        latitude = position.latitude_deg
        longitude = position.longitude_deg
        altitude = position.absolute_altitude_m
        break

    # Flag para verificar se houve movimento
    moved = False
    east = False
    north = False

    # Ajuste na posição com base no deslocamento do frame
    if abs(x_offset) > TOLERANCE:  
        if x_offset > 0:
            longitude += STEP_SIZE  # Mover para direita
        else:
            longitude -= STEP_SIZE  # Mover para esquerda
            east = True
        moved = True

    if abs(y_offset) > TOLERANCE:  
        if y_offset > 0:
            latitude += STEP_SIZE  # Mover para frente
            north = True
        else:
            latitude -= STEP_SIZE  # Mover para trás
        moved = True

    if moved:
        global last_move_time
        if(time.time() - last_move_time >= 5):
        
            # print(f"Movendo para: {latitude}, {longitude}")
            print(f"Ir para: {"Leste" if east else "Oeste"} e {"Norte" if north else "Sul"}")
            await drone.action.goto_location(latitude, longitude, altitude, 0)
            last_move_time = time.time()

    # Retorna True se centralizado
    return abs(x_offset) < TOLERANCE and abs(y_offset) < TOLERANCE

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

            # print(f"{cx}, {cy}, {frame_width}, {frame_height}")

            # Ajusta posição do drone até a figura estar centralizada
            centralizado = await move_to_target(cx, cy, frame_width, frame_height)

        if centralizado:
            await land_drone()
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Executa a lógica assíncrona
asyncio.run(main())
destroy_windows()