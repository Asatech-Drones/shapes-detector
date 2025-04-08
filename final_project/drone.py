import asyncio
import random
from mavsdk import System
import cv2

from shape_detector import cap, detect_shape

# Definições de tolerância para centralização no frame
TOLERANCE = 10  # Pixels
STEP_SIZE = 0.00001  # Pequeno ajuste na latitude/longitude

# Inicializar o drone
drone = System()

async def connect_drone():
    """ Conecta ao drone e realiza a decolagem. """
    await drone.connect(system_address="udp://:14550")
    print("Conectando ao drone...")
    await asyncio.sleep(2)

    await drone.action.arm()
    await drone.action.takeoff()
    print("Decolagem concluída...")

    # Tempo para estabilizar
    await asyncio.sleep(5)

async def detect_figure():
    """ Simula a detecção da figura no frame da câmera. Retorna (cx, cy). """
    # Resolução da câmera (simulada)
    frame_width, frame_height = 640, 480
    
    # Simula uma detecção aleatória da figura geométrica
    cx = random.randint(0, frame_width)
    cy = random.randint(0, frame_height)

    return cx, cy, frame_width, frame_height

async def move_to_target(cx, cy, frame_width, frame_height):
    """ Move o drone para centralizar a figura no frame da câmera. """
    x_offset = cx - frame_width // 2
    y_offset = cy - frame_height // 2
    
    # Obtém posição atual do drone
    async for position in drone.telemetry.position():
        latitude, longitude, altitude = position.latitude_deg, position.longitude_deg, position.absolute_altitude_m
        break

    # Flag para verificar se houve movimento
    moved = False

    # Ajuste na posição com base no deslocamento do frame
    if abs(x_offset) > TOLERANCE:  
        if x_offset > 0:
            longitude += STEP_SIZE  # Mover para direita
        else:
            longitude -= STEP_SIZE  # Mover para esquerda
        moved = True

    if abs(y_offset) > TOLERANCE:  
        if y_offset > 0:
            latitude += STEP_SIZE  # Mover para frente
        else:
            latitude -= STEP_SIZE  # Mover para trás
        moved = True

    if moved:
        print(f"Movendo para: {latitude}, {longitude}")
        await drone.action.goto_location(latitude, longitude, altitude, 0)

    # Retorna True se centralizado
    return abs(x_offset) < TOLERANCE and abs(y_offset) < TOLERANCE

async def land_drone():
    """ Inicia o procedimento de pouso do drone. """
    print("Centralizado! Iniciando pouso...")
    await drone.action.land()

async def main():
    """ Fluxo principal do sistema de navegação autônoma. """
    await connect_drone()

    while True:
        success, frame = cap.read()
        xRect, yRect, wRect, hRect = detect_shape(frame)

        cx = xRect + wRect / 2
        cy = yRect + hRect / 2

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # cx, cy, frame_width, frame_height = await detect_figure()
        # print(f"Figura detectada em: ({cx}, {cy})")

        # Ajusta posição do drone até a figura estar centralizada
        centralizado = await move_to_target(cx, cy, frame_width, frame_height)

        if centralizado:
            await land_drone()
            break  # Fim da missão

# Executa a lógica assíncrona
asyncio.run(main())
