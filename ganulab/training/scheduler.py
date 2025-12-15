from dataclasses import dataclass

@dataclass
class Phase:
    name: str       # "generator" o "discriminator"
    steps: int      # Cuántos pasos consecutivos ejecutar esta fase

class PhaseScheduler:
    """
    Controla el ciclo de entrenamiento para arquitecturas multi-agente (GANs).
    Ejemplo WGAN: 5 pasos de Crítico por 1 paso de Generador.
    """
    def __init__(self, phases: list[Phase]):
        self.phases = phases
        self._phase_idx = 0
        self._step_in_phase = 0

    def get_current_phase(self) -> str:
        """Devuelve el nombre de la fase actual."""
        return self.phases[self._phase_idx].name

    def step(self):
        """Avanza el contador interno. Debe llamarse al final de cada batch."""
        self._step_in_phase += 1
        current_phase_limit = self.phases[self._phase_idx].steps

        if self._step_in_phase >= current_phase_limit:
            # Cambio de fase
            self._step_in_phase = 0
            self._phase_idx = (self._phase_idx + 1) % len(self.phases)

# Ejemplo de uso interno:
# scheduler = PhaseScheduler([
#     Phase("discriminator", 5),
#     Phase("generator", 1)
# ])