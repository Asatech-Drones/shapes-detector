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

# Inicializar o drone
drone = System()
last_move_time = 0.0

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

    # Enviar primeiro comando nulo e iniciar o modo offboard
    print("Iniciando modo Offboard...")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    await drone.offboard.start()
    print("Modo Offboard iniciado com sucesso.")

    await teste_movimento(drone)

    global last_move_time
    last_move_time = time.time()


async def move_to_target(cx, cy, frame_width, frame_height):
    """ Move o drone para centralizar a figura no frame da câmera, 
    usando o referencial do corpo do drone. """
    x_offset = frame_width // 2 - cx
    y_offset = frame_height // 2 - cy

    print(f"Offsets: x = {x_offset}, y = {y_offset}")

    forward = 0.0
    right = 0.0
    moved = False

    # Ajuste de tolerância: só move se deslocamento for significativo
    if abs(x_offset) > TOLERANCE:
        # direita no frame = esquerda no corpo
        right = -0.5 if x_offset > 0 else 0.5  
        moved = True

    if abs(y_offset) > TOLERANCE:
        # baixo no frame = frente no corpo
        forward = -0.5 if y_offset > 0 else 0.5  
        moved = True

    if moved:
        global last_move_time
        if time.time() - last_move_time >= 3:
            print(f"Movendo: forward = {forward}, right = {right}")
            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(forward, right, 0.0, 0.0)
            )
            
            last_move_time = time.time()
    else:
        # Para o drone se estiver centralizado
        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(forward, right, 0.0, 0.0)
        )

    # Retorna True se centralizado
    return abs(x_offset) < TOLERANCE and abs(y_offset) < TOLERANCE


# async def move_to_target(cx, cy, frame_width, frame_height):
#     """ Move o drone para centralizar a figura no frame da câmera. """
#     x_offset = frame_width // 2 - cx
#     y_offset = frame_height // 2 - cy

#     print(f"offsets: {x_offset}, {x_offset}")
    
#     # Obtém posição atual do drone
#     async for position in drone.telemetry.position():
#         latitude = position.latitude_deg
#         longitude = position.longitude_deg
#         altitude = position.absolute_altitude_m
#         break

#     # Flag para verificar se houve movimento
#     moved = False
#     east = False
#     north = False

#     # Ajuste na posição com base no deslocamento do frame
#     if abs(x_offset) > TOLERANCE:  
#         if x_offset > 0:
#             longitude += STEP_SIZE  # Mover para direita
#         else:
#             longitude -= STEP_SIZE  # Mover para esquerda
#             east = True
#         moved = True

#     if abs(y_offset) > TOLERANCE:  
#         if y_offset > 0:
#             latitude += STEP_SIZE  # Mover para frente
#             north = True
#         else:
#             latitude -= STEP_SIZE  # Mover para trás
#         moved = True

#     if moved:
#         global last_move_time
#         if(time.time() - last_move_time >= 5):
        
#             # print(f"Movendo para: {latitude}, {longitude}")
#             print(f"Ir para: {"Leste" if east else "Oeste"} e {"Norte" if north else "Sul"}")
#             await drone.action.goto_location(latitude, longitude, altitude, 0)
#             last_move_time = time.time()

#     # Retorna True se centralizado
#     return abs(x_offset) < TOLERANCE and abs(y_offset) < TOLERANCE

async def land_drone():
    """ Inicia o procedimento de pouso do drone. """
    print("Centralizado! Iniciando pouso...")
    await drone.action.land()

async def main():
    # """ Fluxo principal do sistema de navegação autônoma. """
    await connect_drone()

    while True:

        # await drone.offboard.set_velocity_body(
        #     VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
        # )
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