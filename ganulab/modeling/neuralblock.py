# ganulab/modeling/neuralblock.py

from __future__ import annotations

import inspect
from typing import Any, List, Optional, Type, Union, Dict

import torch
import torch.nn as nn

# Imports del ecosistema ganulab
from ganulab.artifacts.layers import nllib
from ganulab.modeling.neurallayer import NeuralLayer, BayesianLayer, IntervalLayer, BlockInput

class NeuralBlock(NeuralLayer, nn.Sequential):
    """
    Bloque secuencial inteligente que orquesta capas Standard, Bayesianas e Intervalares.
    
    Características:
      - Propagación Híbrida: Maneja tensores puntuales e intervalos (tuplas).
      - Agregación de Pérdidas: Suma automática de KL y Width loss.
      - Constructor 'wire': Factory method para ensamblar arquitecturas complejas fácilmente.
    """

    def __init__(self, *args: nn.Module):
        # -----------------------------------------------------------
        # CORRECCIÓN CRÍTICA DE INICIALIZACIÓN (Diamond Inheritance)
        # -----------------------------------------------------------
        
        # 1. Primero inicializamos nn.Sequential.
        nn.Sequential.__init__(self, *args)
        
        # 2. Registramos manualmente el buffer que NeuralLayer necesita.
        self.register_buffer("_zero", torch.tensor(0.0), persistent=False)

    # ==================================================================
    # 1. Forward Híbrido (Puntual -> Intervalar)
    # ==================================================================
    def forward(self, *inputs: Any) -> BlockInput:
        """
        Ejecuta la secuencia de capas manejando la transición de tipos de datos.
        """
        # Normalización de entrada
        if len(inputs) == 1:
            x: BlockInput = inputs[0]
        elif len(inputs) == 2:
            x = (inputs[0], inputs[1])
        else:
            raise ValueError(f"NeuralBlock espera 1 o 2 tensores de entrada, recibió {len(inputs)}")

        for module in self:
            # --- Caso A: Flujo Puntual (Tensor) ---
            if isinstance(x, torch.Tensor):
                x = module(x)

            # --- Caso B: Flujo Intervalar (Tuple) ---
            elif isinstance(x, tuple) and len(x) == 2:
                lower, upper = x
                
                if isinstance(module, IntervalLayer):
                    # Fallback punto medio si la capa no tiene lógica explícita
                    x_mid = 0.5 * (lower + upper)
                    x = module(x_mid) 
                
                else:
                    # Capa estándar (ej. ReLU, Tanh) aplicada a intervalo
                    # Asumimos monotonía para activaciones simples
                    new_lower = module(lower)
                    new_upper = module(upper)
                    x = (new_lower, new_upper)
            else:
                 raise TypeError(f"Tipo de dato inválido en el flujo: {type(x)}")

        return x

    # ==================================================================
    # 2. Agregación de Pérdidas (Optimizado)
    # ==================================================================
    def kl_loss(self) -> torch.Tensor:
        """Suma la KL divergence de todas las capas Bayesianas hijas."""
        total = self._zero 
        for m in self.modules():
            if isinstance(m, BayesianLayer) and m is not self:
                total = total + m.kl_loss()
        return total

    def width_loss(self, p: float = 2.0) -> torch.Tensor:
        """Suma la Width loss de todas las capas Intervalares hijas."""
        total = self._zero
        for m in self.modules():
            if isinstance(m, IntervalLayer) and m is not self:
                total = total + m.width_loss(p=p)
        return total

    # ==================================================================
    # 3. Factory 'wire' (Constructor Inteligente)
    # ==================================================================
    @classmethod
    def wire(
        cls,
        *,
        # Arquitectura
        layers_type: str | List[str] = "simple.Linear",
        input_layer_type: Optional[str] = None,
        output_layer_type: Optional[str] = None,
        
        # Dimensiones
        input_dimension: int = 1,
        hidden_dimension: int = 1,
        output_dimension: int = 1,
        layers_num: int = 2,
        
        # Modificadores
        activation: str = "LeakyReLU",
        final_activation: Optional[str] = None, # <--- NUEVO: Activación tras la capa de salida
        leakage: float = 0.0,
        dropout: float = 0.0,
        
        # Hiperparámetros Específicos
        sigma_prior: Optional[float] = None,
        init_width: Optional[float] = None,
        
        # Kwargs genéricos
        **layer_kwargs: Any,
    ) -> "NeuralBlock":
        """
        Construye un NeuralBlock ensamblando capas del catálogo `nllib`.
        """

        if layers_num < 2:
            raise ValueError("layers_num debe ser al menos 2 (entrada + salida).")

        # --- A. Preparación de Tipos ---
        hidden_types = [layers_type] if isinstance(layers_type, str) else layers_type
        
        in_type = input_layer_type if input_layer_type else hidden_types[0]
        out_type = output_layer_type if output_layer_type else hidden_types[-1]

        # --- B. Helper de Resolución ---
        def get_class(path: str) -> Type[nn.Module]:
            try:
                family, name = path.split(".")
                return getattr(getattr(nllib, family), name)
            except (ValueError, AttributeError):
                raise ValueError(f"No se pudo resolver la capa '{path}' en nllib.")

        # --- C. Helper de Instanciación Inteligente ---
        def instantiate(type_str: str, in_d: int, out_d: int) -> nn.Module:
            LayerClass = get_class(type_str)
            
            # 1. Inspeccionar firma
            sig = inspect.signature(LayerClass)
            params = sig.parameters
            
            # 2. Construir kwargs
            kwargs = layer_kwargs.copy()
            
            # 3. Inyección de Dimensiones
            if "in_features" in params:
                kwargs["in_features"] = in_d
                kwargs["out_features"] = out_d
            elif "in_channels" in params:
                kwargs["in_channels"] = in_d
                kwargs["out_channels"] = out_d
            else:
                # Intento robusto de instanciación con manejo de error mejorado
                try:
                    return LayerClass(in_d, out_d, **kwargs)
                except TypeError as e:
                    # Detectar si el usuario intentó usar una Activación como Capa
                    if "takes" in str(e) and "positional argument" in str(e):
                        raise TypeError(
                            f"La capa '{type_str}' no acepta dimensiones (in={in_d}, out={out_d}). "
                            f"Si es una función de activación, úsala en 'activation' o 'final_activation'."
                        ) from e
                    raise e

            # 4. Inyección de Hiperparámetros
            if "prior_std" in params and sigma_prior is not None:
                kwargs["prior_std"] = sigma_prior
            elif "sigma_prior" in params and sigma_prior is not None:
                kwargs["sigma_prior"] = sigma_prior
                
            if "init_width" in params and init_width is not None:
                kwargs["init_width"] = init_width
                
            return LayerClass(**kwargs)

        # --- D. Helper de Activación ---
        def get_act(name: str) -> nn.Module:
            full_name = name if "." in name else f"simple.{name}"
            Act = get_class(full_name)
            if "LeakyReLU" in name and leakage > 0:
                return Act(negative_slope=leakage)
            return Act()

        # --- E. Ensamblaje ---
        modules: List[nn.Module] = []
        
        # 1. Capa de Entrada
        modules.append(instantiate(in_type, input_dimension, hidden_dimension))
        
        # 2. Capas Ocultas
        for i in range(layers_num - 2):
            modules.append(get_act(activation))
            if dropout > 0:
                modules.append(nn.Dropout(p=dropout))
            
            layer_type = hidden_types[i % len(hidden_types)]
            modules.append(instantiate(layer_type, hidden_dimension, hidden_dimension))
            
        # 3. Capa de Salida
        modules.append(get_act(activation))
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout))
            
        modules.append(instantiate(out_type, hidden_dimension, output_dimension))

        # 4. Activación Final (Opcional)
        if final_activation:
            modules.append(get_act(final_activation))

        return cls(*modules)