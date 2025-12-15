# ganulab/modeling/neuralmodel.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Callable, Dict, Optional, Type, Union, List, Tuple

# Imports internos
from ganulab.modeling.neurallayer import NeuralLayer, BlockInput
from ganulab.modeling.neuralblock import NeuralBlock
from ganulab.modeling.lossfunction import LossFunction

# ==============================================================================
# 1. Latent Sampler (Generador de Ruido)
# ==============================================================================
class LatentSampler(nn.Module):
    """
    Generador de vectores latentes (ruido) compatible con el dispositivo del modelo.
    Soporta distribuciones Normales y Uniformes.
    """
    def __init__(self, dim: int, distribution: str = "normal"):
        super().__init__()
        self.dim = dim
        self.distribution = distribution.lower()
        
        if self.distribution not in ("normal", "uniform"):
            raise ValueError("LatentSampler solo soporta 'normal' o 'uniform'.")
            
        # Dummy buffer para rastrear el dispositivo del modelo automáticamente
        self.register_buffer("_device_tracker", torch.tensor(0.0), persistent=False)

    def forward(self, batch_size: int) -> torch.Tensor:
        if isinstance(batch_size, torch.Tensor):
            batch_size = int(batch_size.item())
            
        device = self._device_tracker.device
        
        if self.distribution == "normal":
            return torch.randn(batch_size, self.dim, device=device)
        else: # uniform
            return torch.rand(batch_size, self.dim, device=device)
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, distribution={self.distribution}"


# ==============================================================================
# 2. Neural Model (El Artefacto Completo)
# ==============================================================================
class NeuralModel(NeuralLayer):
    """
    Representación completa de una Red Neuronal lista para entrenar.
    
    Incorpora gestión de dispositivo (CPU/GPU) antes de la creación del optimizador.
    """

    def __init__(
        self, 
        *args: nn.Module,
        loss_fn: Optional[LossFunction] = None,
        optimizer_cls: Type[optim.Optimizer] = optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        encoder: Optional[Callable[[Any], BlockInput]] = None,
        decoder: Optional[Callable[[BlockInput], Any]] = None,
        latent_sampler: Optional[LatentSampler] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        device: Union[str, torch.device] = "cpu"  # <--- NUEVO ARGUMENTO
    ):
        # Inicializamos NeuralLayer
        super().__init__()

        # --- CORAZÓN DEL MODELO ---
        self.net = NeuralBlock(*args)

        # --- Componentes "Activos" ---
        self.loss_fn = loss_fn
        
        # --- Componentes Auxiliares ---
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = latent_sampler
        self.input_shape = input_shape

        # --- GESTIÓN DE DISPOSITIVO (CRÍTICO) ---
        # Movemos el modelo al dispositivo ANTES de crear el optimizador.
        # Esto asegura que el optimizador referencie los parámetros en GPU.
        self.to(device)

        # --- Optimizador ---
        opt_kwargs = optimizer_kwargs or {"lr": 1e-3}
        self.optimizer = optimizer_cls(self.parameters(), **opt_kwargs)

    # ==========================================================================
    # Lógica de Forward
    # ==========================================================================
    def forward(self, input_data: Any = None, batch_size: Optional[int] = None) -> Any:
        # 1. Generación Latente
        x = input_data
        if x is None:
            if self.sampler is not None:
                if batch_size is None:
                    raise ValueError("Se requiere 'batch_size' para generar ruido latente.")
                x = self.sampler(batch_size)
            else:
                raise ValueError("Input es None y no hay LatentSampler configurado.")
        
        # 2. Validación
        if input_data is not None and self.input_shape is not None and isinstance(x, torch.Tensor):
            if x.shape[1:] != self.input_shape:
                raise ValueError(f"Shape inválido. Esperado [B, {self.input_shape}], recibido {x.shape}")

        # 3. Encoder
        if self.encoder is not None:
            x = self.encoder(x)

        # 4. NeuralBlock Forward
        if isinstance(x, tuple):
            x = self.net(*x)
        else:
            x = self.net(x)

        # 5. Decoder
        if self.decoder is not None:
            x = self.decoder(x)

        return x

    # ==========================================================================
    # Delegación de Pérdidas
    # ==========================================================================
    def kl_loss(self) -> torch.Tensor:
        return self.net.kl_loss()

    def width_loss(self, p: float = 2.0) -> torch.Tensor:
        return self.net.width_loss(p)

    # ==========================================================================
    # Factory Wire (Unificado)
    # ==========================================================================
    @classmethod
    def wire(
        cls,
        # --- Args para NeuralBlock.wire ---
        layers_type: Union[str, List[str]] = "simple.Linear",
        input_dimension: int = 1,
        output_dimension: int = 1,
        hidden_dimension: int = 32,
        layers_num: int = 3,
        activation: str = "LeakyReLU",
        # --- Args para NeuralModel ---
        loss_fn: Optional[LossFunction] = None,
        optimizer_cls: Type[optim.Optimizer] = optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        encoder: Optional[Callable] = None,
        decoder: Optional[Callable] = None,
        latent_dim: Optional[int] = None, 
        input_shape: Optional[Tuple[int, ...]] = None,
        device: Union[str, torch.device] = "cpu", # <--- NUEVO ARGUMENTO EN WIRE
        # --- Kwargs extra ---
        **kwargs
    ) -> "NeuralModel":
        
        # 1. Construimos el bloque temporal
        temp_block = NeuralBlock.wire(
            layers_type=layers_type,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            hidden_dimension=hidden_dimension,
            layers_num=layers_num,
            activation=activation,
            **kwargs 
        )
        
        layers = list(temp_block.children())

        # 2. Configurar Sampler
        sampler = None
        if latent_dim is not None:
            sampler = LatentSampler(dim=latent_dim)

        # 3. Instanciar NeuralModel (Pasando el device)
        model = cls(
            *layers,
            loss_fn=loss_fn,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            encoder=encoder,
            decoder=decoder,
            latent_sampler=sampler,
            input_shape=input_shape,
            device=device  # <--- SE PASA AL CONSTRUCTOR
        )
        
        return model

    # ==========================================================================
    # Utilities
    # ==========================================================================
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
        self.train()
        return out
    
# ==============================================================================
# 3. Model Inspection Utility
# ==============================================================================
def inspect_model(model: NeuralModel) -> str:
    """
    Genera un informe detallado y estético sobre la arquitectura y configuración 
    de un NeuralModel.
    
    Analiza:
      - Arquitectura capa por capa (Tipos, Shapes, Priors).
      - Recuento de parámetros (Trainable vs Fixed).
      - Configuración de entrenamiento (Optimizador, LR, Device).
      - Función de Pérdida compuesta.
      - Componentes auxiliares (Encoder, Decoder, Sampler).
    
    Returns:
        str: El reporte completo formateado.
    """
    
    lines = []
    def log(s=""): lines.append(s)
    
    # --- Header ---
    log("\n" + "━" * 60)
    log(f"🔍 NEURAL MODEL INSPECTION REPORT")
    log("━" * 60)
    
    # --- 1. General Info ---
    device_type = next(model.parameters()).device.type if list(model.parameters()) else "cpu"
    log(f"📍 Device      : {device_type.upper()}")
    log(f"🧠 Class       : {model.__class__.__name__}")
    
    # --- 2. Architecture & Parameters ---
    log("\n🏗️  ARCHITECTURE")
    log("┌" + "─"*58 + "┐")
    log(f"│ {'Layer Type':<20} | {'Shape / Config':<33} │")
    log("├" + "─"*58 + "┤")
    
    total_params = 0
    trainable_params = 0
    
    # Analizamos el bloque interno (self.net)
    for idx, layer in enumerate(model.net):
        layer_name = layer.__class__.__name__
        config_str = ""
        
        # Extracción de info específica
        if hasattr(layer, 'extra_repr'):
            # Limpiamos el extra_repr para que quepa
            info = layer.extra_repr().replace('\n', ' ')
            # Intentamos extraer dimensiones clave
            if 'in_features' in info:
                # Linear
                in_f = getattr(layer, 'in_features', '?')
                out_f = getattr(layer, 'out_features', '?')
                config_str = f"In:{in_f} ➜ Out:{out_f}"
                
                # Detalles Bayesianos
                if hasattr(layer, 'prior_init_value'):
                    config_str += f" [Prior:{layer.prior_init_value}]"
                # Detalles Intervalares
                if hasattr(layer, 'init_width'):
                    config_str += f" [Width:{layer.init_width}]"
                    
            elif 'in_channels' in info:
                # Conv
                in_c = getattr(layer, 'in_channels', '?')
                out_c = getattr(layer, 'out_channels', '?')
                k = getattr(layer, 'kernel_size', '?')
                config_str = f"In:{in_c} ➜ Out:{out_c} (k={k})"
            else:
                # Activaciones / Otros
                if "NegativeSlope" in layer_name or "Leaky" in layer_name:
                    slope = getattr(layer, 'negative_slope', 0.01)
                    config_str = f"slope={slope}"
                elif "Dropout" in layer_name:
                    p = getattr(layer, 'p', 0.5)
                    config_str = f"p={p}"
        
        log(f"│ {idx:02d}. {layer_name:<16} | {config_str:<33} │")
        
        # Conteo de parámetros
        for p in layer.parameters():
            p_count = p.numel()
            total_params += p_count
            if p.requires_grad:
                trainable_params += p_count

    log("└" + "─"*58 + "┘")
    
    # --- 3. Parameter Summary ---
    log(f"\n📊 PARAMETERS")
    log(f"   • Total      : {total_params:,}")
    log(f"   • Trainable  : {trainable_params:,}")
    log(f"   • Non-trainable: {total_params - trainable_params:,}")

    # --- 4. Training Config ---
    log("\n⚙️  TRAINING CONFIG")
    
    # Optimizer
    opt_name = model.optimizer.__class__.__name__
    opt_defaults = model.optimizer.defaults
    lr = opt_defaults.get('lr', '?')
    wd = opt_defaults.get('weight_decay', 0)
    log(f"   • Optimizer  : {opt_name}")
    log(f"   • Learning Rate: {lr}")
    if wd > 0:
        log(f"   • Weight Decay : {wd}")

    # Loss Function
    if model.loss_fn:
        log(f"\n   • Loss Function Composition:")
        lf = model.loss_fn
        if lf.main:
            log(f"     [Main] {lf.main.name:<20} (w={lf.main.weight})")
        for aux in lf.aux:
            log(f"     [Aux ] {aux.name:<20} (w={aux.weight})")
        for pen in lf.penalties:
            log(f"     [Pen ] {pen.name:<20} (w={pen.weight})")
    else:
        log(f"\n   • Loss Function: None (External)")

    # --- 5. Auxiliaries ---
    if model.sampler or model.encoder or model.decoder:
        log("\n🧩 AUXILIARIES")
        if model.sampler:
            log(f"   • Latent Sampler : {model.sampler.extra_repr()}")
        if model.encoder:
            enc_name = getattr(model.encoder, "__name__", str(model.encoder))
            log(f"   • Encoder        : {enc_name}")
        if model.decoder:
            dec_name = getattr(model.decoder, "__name__", str(model.decoder))
            log(f"   • Decoder        : {dec_name}")

    log("\n" + "━" * 60)
    
    report = "\n".join(lines)
    print(report)
    return report