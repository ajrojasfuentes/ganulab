from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ganulab.modeling.neuralmodel import NeuralModel
from ganulab.training.limitations import Limitation
from ganulab.training.scheduler import PhaseScheduler

class TrainingStrategy(ABC):
    """
    Define la lógica matemática de un paso de entrenamiento (Step).
    Conecta: Model + Loss + Optimizer + Limitations + Scheduler.
    """
    def __init__(
        self, 
        limitations: Optional[List[Limitation]] = None,
        scheduler: Optional[PhaseScheduler] = None
    ):
        self.limitations = limitations or []
        self.scheduler = scheduler

    def on_epoch_start(self, model: NeuralModel) -> None:
        """Hook opcional al inicio de cada época."""
        pass

    @abstractmethod
    def training_step(
        self, 
        model: NeuralModel, 
        batch: Any, 
        batch_idx: int
    ) -> Dict[str, float]:
        """
        Ejecuta Forward -> Loss -> Backward -> Optimizer -> Limitations.
        Debe devolver un diccionario de métricas (ej: {'loss': 0.5}).
        """
        pass

# ==============================================================================
# Estrategia Supervisada (Estándar)
# ==============================================================================

class SupervisedStrategy(TrainingStrategy):
    """
    Estrategia clásica: Forward -> Loss -> Backward -> Step.
    """
    def training_step(
        self, 
        model: NeuralModel, 
        batch: Any, 
        batch_idx: int
    ) -> Dict[str, float]:
        
        # 0. Preparar datos
        # Asumimos que batch es (inputs, targets) o similar
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
            x, y = x.to(model.device), y.to(model.device)
            kwargs = {"target": y}
        else:
            # Caso unsupervised o custom batch
            x = batch.to(model.device)
            kwargs = {}

        # 1. Limpiar gradientes
        if model.optimizer:
            model.optimizer.zero_grad()

        # 2. Forward
        # NeuralModel.__call__ delega a NeuralModule.forward
        pred = model(x)

        # 3. Loss Calculation
        # Pasamos pred, inputs y targets (via kwargs) a la LossFunction inteligente
        loss = model.loss_fn(pred=pred, input=x, **kwargs)

        # 4. Backward
        loss.backward()

        # 5. Limitations (Pre-Step): Gradient Clipping
        for limit in self.limitations:
            # Tip: Podríamos añadir un flag a Limitation para saber si es pre/post
            # Por ahora aplicamos todas aquí, asumiendo que WeightClipping 
            # no daña nada si se aplica antes (aunque lo ideal es post).
            # Para mayor control, podríamos dividir self.limitations en pre_limits y post_limits.
            limit.apply(model)

        # 6. Optimizer Step
        if model.optimizer:
            model.optimizer.step()
            
        # 7. Limitations (Post-Step): Weight Clipping (si existiera lógica separada)
        # (Aquí volveríamos a llamar a las limitaciones de tipo peso)

        return {"loss": loss.item()}