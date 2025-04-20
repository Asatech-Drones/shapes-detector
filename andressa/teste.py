""" 
Este módulo simula um controle de drone passando por traves coloridas (hastes).
A lógica consiste em detectar traves em cores específicas (vermelho, azul, verde) 
e orientar o drone a passar por elas alternando o lado (esquerda/direita) conforme a ordem definida.

Funcionalidades:
- Detecção de cores (hastes) utilizando a técnica de contornos em imagens.
- Controle da ordem das traves a serem passadas.
- Exibição de indicações visuais para o drone no vídeo.
"""
import cv2
from mission import Mission
from drone import Drone 
from haste_detector import HasteDetector

# define o intervalo de cores em HSV para as hastes
def main():
    mission = Mission(sequence=['yellow_green','pink_pen','blue_pen'], initial_side='left')
    drone = Drone(mission)
    haste_detector = HasteDetector()
    
    # def on_mouse(event, x, y, flags, param):
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         hsv = cv2.cvtColor(param, cv2.COLOR_BGR2HSV)
    #         pixel = hsv[y, x]
    #         print(f"HSV em ({x},{y}): {pixel}")  # [H, S, V]

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        # cv2.imshow("Clique para pegar HSV", frame)
        # cv2.setMouseCallback("Clique para pegar HSV", on_mouse, frame)
    
        # Obtém a largura do frame
        frame_width = frame.shape[1]

        if mission.is_mission_completed():
            cv2.putText(frame, "Missao Concluida!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.imshow("Missao Slalom", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        target_color = mission.sequence[mission.current_index]
        frame, detections, mask = haste_detector.detect_staves(frame, target_color, frame_width)

        # Passando frame_width para a função determine_closest_haste
        if detections:
            closest = haste_detector.determine_closest_haste(detections, frame_width)
            center_x = closest['x'] + closest['w'] // 2
            center_y = closest['y'] + closest['h'] // 2
            haste_detector.update((center_x, center_y))
            haste_detector.visible_last_frame = True
        else:
            predicted = haste_detector.update(None)
            predicted_x, predicted_y = int(predicted[0]), int(predicted[1])
            closest = {'x': predicted_x - 10, 'y': predicted_y - 20, 'w': 20, 'h': 40}

            # se a haste foi visível no frame anterior e agora não está, consideramos como "ultrapassada"
            if haste_detector.visible_last_frame:
                drone_side = drone.decide_side(frame_width, closest)

                if drone_side == mission.current_side:
                    print(f"Trave {mission.current_index + 1} ({target_color}) percorrida corretamente pelo {mission.current_side}")
                else:
                    print(f"Trave {mission.current_index + 1} ({target_color}) percorrida pelo lado errado ({drone_side})")

                mission.passed_staves.append(target_color)
                mission.update_side()

            haste_detector.visible_last_frame = False  # reset


        if closest:
            drone_side = drone.decide_side(frame_width, closest)
            drone.display_instruction(frame)

            if haste_detector.was_visible and len(detections) == 0:
                print(f"Trave {mission.current_index + 1} ({target_color}) percorrida pelo {mission.current_side}")
                mission.passed_staves.append(target_color)
                mission.update_side()

                # Atualiza status
                haste_detector.was_visible = len(detections) > 0
                
            if drone_side == mission.current_side and haste_detector.was_visible and len(detections) == 0:
                print(f"Trave {mission.current_index + 1} ({target_color}) percorrida corretamente pelo {mission.current_side}")
                mission.passed_staves.append(target_color)
                mission.update_side()


        else:
            drone.display_instruction(frame)
            cv2.putText(frame, f"Alvo: {target_color.upper()}", (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Exibe o frame original e a máscara
        cv2.imshow("Frame Original", frame)
        cv2.imshow("Mascara", mask)  # Exibe a máscara gerada

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()