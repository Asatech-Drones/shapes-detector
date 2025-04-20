import cv2

class Drone:
    def __init__(self, mission):
        self.mission = mission

    def decide_side(self, frame_width, closest_haste):
        """Decide de que lado o drone deve passar (esquerda/direita)."""
        if closest_haste:
            center_x = closest_haste['x'] + closest_haste['w'] // 2
            position = 'left' if center_x < frame_width // 2 else 'right'
            return position
        return None

    def display_instruction(self, frame):
        """Exibe no vídeo as instruções do lado correto."""
        cv2.putText(frame, f"Passe pelo lado: {self.mission.current_side.upper()}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)