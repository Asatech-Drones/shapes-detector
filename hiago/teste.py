from pymavlink.dialects.v20 import common as mavlink2
import serial
import time

# Cria objeto serial
port = serial.Serial('/dev/ttyUSB0', baudrate=57600)

# Cria sistema MAVLink
mav = mavlink2.MAVLink(port)
mav.srcSystem = 1

def zigzag_scan(rows, cols):
    path = []
    for row in range(rows):
        if row % 2 == 0:
            for col in range(cols):
                path.append((row, col))
        else:
            for col in reversed(range(cols)):
                path.append((row, col))
    return path

rows = 5
cols = 4
trajetoria = zigzag_scan(rows, cols)

for ponto in trajetoria:
    x, y = ponto
    # Exemplo: envia como coordenada local (posição fictícia)
    msg = mav.local_position_ned_encode(
        int(time.time() * 1e6),  # tempo em microssegundos
        1, # ID do sistema (pode ser 1 para o emissor)
        1, # ID do componente (por exemplo, o "componente GPS")
        float(x),  # posição x
        float(y),  # posição y
        0.0,        # posição z 
        0.0, 0.0, 0.0  # velocidades
    )
    port.write(msg.pack(mav))
    print(f"Enviado: ({x}, {y})")
    time.sleep(0.2)