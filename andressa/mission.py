class Mission:
    def __init__(self, sequence, initial_side):
        self.sequence = sequence  # ordem das hastes
        self.current_index = 0  # controla a trave atual
        self.current_side = initial_side  # lado atual (esquerda/direita)
        self.passed_staves = []  # lista de traves já passadas

    def update_side(self):
        """Alterna o lado após a travessia de uma trave."""
        self.current_side = 'right' if self.current_side == 'left' else 'left'
        self.current_index += 1

    def is_mission_completed(self):
        """Verifica se a missão foi concluída."""
        return self.current_index >= len(self.sequence)