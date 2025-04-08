import asyncio
from mavsdk import System

async def fly_routine():
    drone = System()
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

    # Espera estabilizar
    await asyncio.sleep(15)

    print("Movendo 5m para frente e subindo 2m...")
    await drone.action.goto_location(47.3977, 8.5456, 2, 0)
    await asyncio.sleep(15)

    print("Movendo para a direita...")
    await drone.action.goto_location(47.3980, 8.5460, 2, 0)
    await asyncio.sleep(15)

    print("Voltando...")
    await drone.action.goto_location(47.3977, 8.5456, 2, 0)
    await asyncio.sleep(15)

    print("Pousando...")
    await drone.action.land()

asyncio.run(fly_routine())
