from abc import ABC, abstractmethod
from typing import Union, Iterable, List
import torch
import torch.nn as nn

# Tipado para evitar importación circular, asumimos que recibirá el NeuralModel o el Module
# En runtime recibirá el NeuralModel.
from typing import Any 

class Limitation(ABC):
    """
    Clase base para restricciones externas aplicadas durante el entrenamiento.
    A diferencia de las Loss Functions, estas modifican directamente los parámetros
    o los gradientes del modelo.
    """
    
    @abstractmethod
    def apply(self, model: Any) -> None:
        """
        Aplica la restricción al modelo.
        
        :param model: Instancia de NeuralModel (o nn.Module).
        """
        pass

# ==============================================================================
# 1. Restricciones de Gradiente (Se aplican ANTES de optimizer.step())
# ==============================================================================

class GradientClipping(Limitation):
    """
    Recorta la norma o el valor de los gradientes para prevenir explosión (Exploding Gradients).
    Fundamental en RNNs y GANs inestables.
    
    Uso:
        lim = GradientClipping(max_norm=1.0)
    """
    def __init__(
        self, 
        max_norm: float = 0.0, 
        clip_value: float = 0.0, 
        norm_type: float = 2.0
    ):
        """
        :param max_norm: Si > 0, recorta la norma global del gradiente a este valor.
        :param clip_value: Si > 0, recorta cada gradiente individualmente a [-val, val].
        :param norm_type: Tipo de norma (generalmente 2.0 para L2).
        
        Nota: Usar max_norm O clip_value, no ambos a la vez (prioridad a max_norm).
        """
        self.max_norm = float(max_norm)
        self.clip_value = float(clip_value)
        self.norm_type = float(norm_type)

    def apply(self, model: Any) -> None:
        # Extraemos el nn.Module del NeuralModel
        module = model.module if hasattr(model, "module") else model
        parameters = [p for p in module.parameters() if p.grad is not None]

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                parameters, 
                max_norm=self.max_norm, 
                norm_type=self.norm_type
            )
        elif self.clip_value > 0:
            torch.nn.utils.clip_grad_value_(
                parameters, 
                clip_value=self.clip_value
            )

# ==============================================================================
# 2. Restricciones de Peso (Se aplican DESPUÉS de optimizer.step())
# ==============================================================================

class WeightClipping(Limitation):
    """
    Fuerza a los pesos a permanecer dentro de un rango [-c, c].
    Es el requisito central de la Wasserstein GAN (WGAN) original para 
    imponer la restricción de Lipschitz-1.
    """
    def __init__(self, lower: float = -0.01, upper: float = 0.01):
        self.lower = float(lower)
        self.upper = float(upper)

    def apply(self, model: Any) -> None:
        module = model.module if hasattr(model, "module") else model
        
        # Iteramos sobre todos los parámetros y aplicamos clamp directamente en la data
        for p in module.parameters():
            p.data.clamp_(self.lower, self.upper)

class SpectralNorm(Limitation):
    """
    Nota: La Normalización Espectral suele aplicarse como un wrapper de capa
    (torch.nn.utils.spectral_norm), pero a veces se implementa como un hook
    post-step. 
    
    Por ahora, lo dejamos como placeholder o para implementaciones custom 
    que requieran re-normalizar pesos manualmente tras el update.
    """
    def apply(self, model: Any) -> None:
        # Placeholder para lógica futura si se requiere SN manual
        pass

# ==============================================================================
# 3. Composición
# ==============================================================================

class ComposeLimitations(Limitation):
    """
    Permite encadenar múltiples limitaciones.
    """
    def __init__(self, limitations: List[Limitation]):
        self.limitations = limitations

    def apply(self, model: Any) -> None:
        for limit in self.limitations:
            limit.apply(model)