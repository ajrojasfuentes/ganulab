from abc import ABC, abstractmethod
from typing import Dict, Any

class StopCriterion(ABC):
    """
    Clase base para decidir si el entrenamiento debe detenerse.
    """
    @abstractmethod
    def should_stop(self, state: Dict[str, Any]) -> bool:
        """
        Recibe el estado actual del entrenamiento (epoch, step, losses, etc.)
        y devuelve True si se debe abortar.
        """
        pass

class MaxEpochs(StopCriterion):
    def __init__(self, max_epochs: int):
        self.max_epochs = max_epochs

    def should_stop(self, state: Dict[str, Any]) -> bool:
        current_epoch = state.get("epoch", 0)
        return current_epoch >= self.max_epochs

class MaxSteps(StopCriterion):
    def __init__(self, max_steps: int):
        self.max_steps = max_steps

    def should_stop(self, state: Dict[str, Any]) -> bool:
        current_step = state.get("global_step", 0)
        return current_step >= self.max_steps

class EarlyStopping(StopCriterion):
    """
    Detiene si una métrica (ej: 'val_loss') no mejora en 'patience' épocas.
    """
    def __init__(self, monitor: str = "loss", patience: int = 5, mode: str = "min"):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.counter = 0

    def should_stop(self, state: Dict[str, Any]) -> bool:
        current_metrics = state.get("metrics", {})
        if self.monitor not in current_metrics:
            return False # Si no existe la métrica, seguimos

        val = current_metrics[self.monitor]
        
        improved = (val < self.best_value) if self.mode == "min" else (val > self.best_value)
        
        if improved:
            self.best_value = val
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            print(f"[EarlyStopping] Deteniendo: {self.monitor} no mejoró en {self.patience} pasos.")
            return True
            
        return False

class ComposeCriteria(StopCriterion):
    """Permite combinar varios criterios (OR lógico)."""
    def __init__(self, *criteria: StopCriterion):
        self.criteria = criteria

    def should_stop(self, state: Dict[str, Any]) -> bool:
        # Si CUALQUIERA dice stop, paramos
        return any(c.should_stop(state) for c in self.criteria)