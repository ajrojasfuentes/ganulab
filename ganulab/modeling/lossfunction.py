# ganulab/modeling/lossfunction.py

from __future__ import annotations
from typing import Callable, List, Optional, Any, Set, Dict, Union
import inspect
import torch

Tensor = torch.Tensor
LossFn = Callable[..., Tensor]

class _LossTerm:
    """
    Clase interna optimizada para almacenar una función de pérdida, su peso y firma.
    Usa __slots__ para reducir overhead de memoria y acceso.
    """
    __slots__ = ('weight', 'fn', 'name', 'required_args', 'all_args')

    def __init__(self, weight: float, fn: LossFn, name: str = ""):
        self.weight = float(weight)
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "unknown_loss")
        
        # --- INTROSPECCIÓN (Solo una vez al inicio) ---
        try:
            sig = inspect.signature(fn)
        except ValueError:
            # Fallback para funciones builtin o C-extensions que no tienen firma
            # Asumimos que aceptan **kwargs si falla la inspección
            self.required_args = set()
            self.all_args = set() 
        else:
            self.required_args = set()
            self.all_args = set()
            
            for param_name, param in sig.parameters.items():
                # Ignoramos *args y **kwargs
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                
                self.all_args.add(param_name)
                if param.default is inspect.Parameter.empty:
                    self.required_args.add(param_name)

    def __call__(self, available_kwargs: Dict[str, Any]) -> Tensor:
        """
        Ejecuta la función inyectando solo los argumentos necesarios.
        Optimizado para el bucle de entrenamiento.
        """
        # 1. Validación rápida (Set operations son muy rápidas en Python)
        if self.required_args:
            missing = self.required_args - available_kwargs.keys()
            if missing:
                raise TypeError(
                    f"Faltan argumentos requeridos para '{self.name}': {missing}. "
                    f"Disponibles: {list(available_kwargs.keys())}"
                )

        # 2. Filtrado de argumentos
        # Si all_args está vacío (ej: función sin argumentos o **kwargs genérico mal detectado),
        # pasamos todo por seguridad, o nada si sabemos que no pide nada.
        # Aquí asumimos el comportamiento seguro: solo pasar lo que pide explícitamente.
        if self.all_args:
            call_kwargs = {
                k: v for k, v in available_kwargs.items() 
                if k in self.all_args
            }
            val = self.fn(**call_kwargs)
        else:
            # Caso raro: función sin argumentos explícitos o que usa **kwargs ciegamente.
            # Intento de llamada con todo (si la función acepta **kwargs) o vacía.
            try:
                val = self.fn(**available_kwargs)
            except TypeError:
                val = self.fn()

        # 3. Validación y Reducción
        if not isinstance(val, torch.Tensor):
            # Intento de conversión automática (ej: si devuelve float)
            val = torch.tensor(val, device=available_kwargs.get('device', 'cpu'))

        # Reducción a escalar si es necesario (Mean reduction por defecto)
        if val.dim() > 0:
            val = val.mean()
            
        return val * self.weight


class LossFunction:
    """
    Orquestador de Funciones de Pérdida Compuestas.
    
    Formula:
        L = ( w_main * L_main + sum(w_aux * L_aux) + sum(w_pen * L_pen) + eta ) / rho
    """

    def __init__(self, eta: float = 0.0, rho: float = 1.0):
        self.main: Optional[_LossTerm] = None
        self.aux: List[_LossTerm] = []
        self.penalties: List[_LossTerm] = []
        
        self.eta = float(eta)
        self.rho = float(rho)
        if abs(self.rho) < 1e-8: # Protección contra división por cero
            raise ValueError("rho no puede ser 0 (o muy cercano a 0).")

    # ---------------------------
    # Configuración (Fluent Interface)
    # ---------------------------
    def set_main(self, weight: float, fn: LossFn) -> "LossFunction":
        self.main = _LossTerm(weight, fn, name="MainLoss")
        return self

    def add_aux(self, weight: float, fn: LossFn) -> "LossFunction":
        self.aux.append(_LossTerm(weight, fn))
        return self

    def add_penalty(self, weight: float, fn: LossFn) -> "LossFunction":
        self.penalties.append(_LossTerm(weight, fn))
        return self

    def set_eta(self, eta: float) -> "LossFunction":
        self.eta = float(eta)
        return self

    def set_rho(self, rho: float) -> "LossFunction":
        if abs(rho) < 1e-8:
            raise ValueError("rho no puede ser 0.")
        self.rho = float(rho)
        return self

    # ---------------------------
    # Evaluación
    # ---------------------------
    def calc(self, **kwargs: Any) -> Tensor:
        """
        Calcula la pérdida total compuesta.
        
        Argumentos típicos en kwargs:
          - outputs (predicciones)
          - targets (etiquetas)
          - model (para penalizaciones KL/Width)
          - inputs (para gradient penalty)
        """
        if self.main is None:
            raise RuntimeError("Main loss no configurada. Usa .set_main() primero.")

        # 1. Main Loss
        total = self.main(kwargs)

        # 2. Auxiliares
        for term in self.aux:
            total = total + term(kwargs)

        # 3. Penalties
        for term in self.penalties:
            total = total + term(kwargs)

        # 4. Constantes globales
        if self.eta != 0.0:
            total = total + self.eta
            
        if self.rho != 1.0:
            total = total / self.rho
            
        return total

    def __call__(self, **kwargs: Any) -> Tensor:
        return self.calc(**kwargs)
    
    def __repr__(self) -> str:
        s = f"LossFunction(eta={self.eta}, rho={self.rho})\n"
        if self.main:
            s += f"  [Main] {self.main.name} (w={self.main.weight})\n"
        if self.aux:
            s += "  [Aux]\n" + "\n".join([f"    + {t.name} (w={t.weight})" for t in self.aux]) + "\n"
        if self.penalties:
            s += "  [Penalty]\n" + "\n".join([f"    + {t.name} (w={t.weight})" for t in self.penalties])
        return s