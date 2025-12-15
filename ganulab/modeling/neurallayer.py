# ganulab/modeling/neurallayer.py

# ======================================================
# Standard library imports
# ======================================================
import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

# ======================================================
# Third-party library imports
# ======================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg 
import torch.distributions as dist
from torch import Tensor

# ======================================================
# Type aliases
# ======================================================
Interval = Tuple[Tensor, Tensor]
BlockInput = Union[Tensor, Interval]


# ▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰
# NeuralLayer
# ▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰
# Description: Base abstract class for all neural layers.
#              Provides common interfaces for auxiliary losses.
# Type: Superclass (Abstract)
# ▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰

class NeuralLayer(nn.Module, ABC):
    """
    Base superclass for neural network layers (Dense, Conv, Bayesian, Interval, etc).
    
    Objectives:
      - Provide a common interface for kl_loss and width_loss.
      - Allow treating different layers as the same abstract type.
      - Defaults to returning 0 for extra losses if not implemented.
    """
    
    def __init__(self):
        super().__init__()
        # OPTIMIZATION: Register a non-persistent buffer for the zero scalar.
        # 1. Handles device movement automatically (model.to(device)).
        # 2. Handles dtype conversion automatically (model.double()).
        # 3. persistent=False ensures it is NOT saved in the state_dict checkpoint.
        # 4. JIT-friendly (no iterators or try-except blocks).
        self.register_buffer("_zero", torch.tensor(0.0), persistent=False)

    def kl_loss(self) -> Tensor:
        return self._zero

    def width_loss(self, p: float = 2.0) -> Tensor:
        return self._zero


# ▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰
# BayesianLayer
# ▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰
# Description: Abstract class implementing Variational Inference logic.
#              Manages weight sampling (mu, rho) and KL divergence.
# Type: Abstract Class
# ▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰

# ======================================================
# JIT Functions
# ======================================================
@torch.jit.script
def reparameterize(mu: Tensor, rho: Tensor, eps: Tensor) -> Tensor:
    """
    Realiza el 'reparameterization trick' de forma fusionada y optimizada.
    w = mu + log(1 + exp(rho)) * eps
    """
    sigma = F.softplus(rho)
    return mu + sigma * eps

# ======================================================
# Class Definition
# ======================================================

class BayesianLayer(NeuralLayer, ABC):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        weight_shape: Tuple[int, ...],
        bias: bool = True,
        prior_std: float = 1.0,
        learnable_prior: bool = False,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight_shape = weight_shape
        self.prior_init_value = float(prior_std) # Guardado para extra_repr
        self.learnable_prior = learnable_prior

        # ---------------- Pesos (W) ---------------- #
        self.weight_mu = nn.Parameter(torch.empty(*weight_shape))
        self.weight_rho = nn.Parameter(torch.empty(*weight_shape))
        
        # Buffer de ruido fijo (útil para predicciones deterministas o debugging)
        self.register_buffer("weight_eps", torch.zeros(*weight_shape))

        # ---------------- Prior ---------------- #
        prior_init = torch.full(weight_shape, self.prior_init_value)
        if learnable_prior:
            self.prior_std = nn.Parameter(prior_init)
        else:
            self.register_buffer("prior_std", prior_init)

        # ---------------- Bias (b) ---------------- #
        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_dim))
            self.bias_rho = nn.Parameter(torch.empty(out_dim))
            self.register_buffer("bias_eps", torch.zeros(out_dim))
            
            bias_prior_init = torch.full((out_dim,), self.prior_init_value)
            if learnable_prior:
                self.bias_prior_std = nn.Parameter(bias_prior_init)
            else:
                self.register_buffer("bias_prior_std", bias_prior_init)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
            self.register_parameter('bias_eps', None)
            self.register_parameter('bias_prior_std', None)

        # ---------------- Inicialización ---------------- #
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Reinicializa los parámetros (mu, rho) a sus valores iniciales.
        """
        # 1. Pesos: Kaiming para mu, constante pequeña para rho
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
            self.weight_rho.fill_(-5.0) # sigma inicial ≈ 0.0067

        # 2. Bias
        if self.bias_mu is not None and self.bias_rho is not None:
            fan_in = self._compute_fan_in(self.weight_shape)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
            with torch.no_grad():
                self.bias_mu.uniform_(-bound, bound)
                self.bias_rho.fill_(-5.0)

    # -----------------------------------------------------
    # String Representation
    # -----------------------------------------------------
    
    def extra_repr(self) -> str:
        s = f'in_dim={self.in_dim}, out_dim={self.out_dim}'
        s += f', prior_std={self.prior_init_value}'
        if self.learnable_prior:
            s += ' (learnable)'
        if self.bias_mu is not None:
            s += ', bias=True'
        return s

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------

    @staticmethod
    def _compute_fan_in(weight_shape: Tuple[int, ...]) -> int:
        if len(weight_shape) < 2: return 1
        return weight_shape[1] * (math.prod(weight_shape[2:]) if len(weight_shape) > 2 else 1)

    @staticmethod
    def _softplus_rho(rho: Tensor) -> Tensor:
        return F.softplus(rho)

    # -----------------------------------------------------
    # Sampling (Usando JIT)
    # -----------------------------------------------------

    def sample_weight(self, use_stored_eps: bool = False) -> Tensor:
        if use_stored_eps:
            eps = self.weight_eps
        else:
            eps = torch.randn_like(self.weight_mu)
        
        # Llamada a la función optimizada JIT
        return reparameterize(self.weight_mu, self.weight_rho, eps)

    def sample_bias(self, use_stored_eps: bool = False) -> Optional[Tensor]:
        if self.bias_mu is None or self.bias_rho is None:
            return None
        
        if use_stored_eps:
            eps = self.bias_eps
        else:
            eps = torch.randn_like(self.bias_mu)
            
        return reparameterize(self.bias_mu, self.bias_rho, eps)

    # -----------------------------------------------------
    # Posterior Views
    # -----------------------------------------------------

    def get_weight_posterior(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.weight_mu, self.weight_rho, self.prior_std

    def get_bias_posterior(self) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
        if self.bias_mu is None or self.bias_rho is None or self.bias_prior_std is None:
            return None
        return self.bias_mu, self.bias_rho, self.bias_prior_std

    # -----------------------------------------------------
    # KL Loss
    # -----------------------------------------------------

    def _compute_kl_term(self, mu: Tensor, rho: Tensor, prior_std: Tensor) -> Tensor:
        # Nota: No usamos JIT aquí porque torch.distributions es complejo.
        sigma = self._softplus_rho(rho)
        q = dist.Normal(mu, sigma)
        p = dist.Normal(torch.zeros_like(mu), prior_std)
        return dist.kl_divergence(q, p).sum()

    def kl_loss(self) -> Tensor:
        kl_w = self._compute_kl_term(self.weight_mu, self.weight_rho, self.prior_std)
        
        if self.bias_mu is not None and self.bias_rho is not None and self.bias_prior_std is not None:
            kl_b = self._compute_kl_term(self.bias_mu, self.bias_rho, self.bias_prior_std)
            return kl_w + kl_b
        
        return kl_w

    # -----------------------------------------------------
    # Forward (Abstract)
    # -----------------------------------------------------

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


# ▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰
# IntervalLayer
# ▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰
# Description: Abstract class implementing Center-Radius Interval Arithmetic.
#              Handles radius parametrization and width regularization.
# Type: Abstract Class
# ▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰

# ======================================================
# JIT Functions
# ======================================================
@torch.jit.script
def split_pos_neg(x: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Optimized split of x into positive and negative parts.
    x_pos = max(x, 0)
    x_neg = min(x, 0)
    """
    x_pos = F.relu(x)
    # -F.relu(-x) is equivalent to min(x, 0) but leverages relu optimization
    x_neg = -F.relu(-x)
    return x_pos, x_neg


# ======================================================
# Class Definition
# ======================================================

class IntervalLayer(NeuralLayer, ABC):
    """
    Generic superclass for interval layers (Center-Radius).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        weight_shape: Tuple[int, ...],
        *,
        bias: bool = True,
        init_width: float = 1e-2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.init_width = float(init_width)
        self.weight_shape = weight_shape 

        # ---------------- Parameters ---------------- #
        self.weight_center = nn.Parameter(torch.empty(*weight_shape))
        self.weight_radius_raw = nn.Parameter(torch.empty(*weight_shape))

        if bias:
            self.bias_center = nn.Parameter(torch.empty(out_dim))
            self.bias_radius_raw = nn.Parameter(torch.empty(out_dim))
        else:
            self.register_parameter('bias_center', None)
            self.register_parameter('bias_radius_raw', None)

        # ---------------- Initialization ---------------- #
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Re-initializes parameters. Standard PyTorch interface.
        """
        # 1. Kaiming init for center (weights)
        nn.init.kaiming_uniform_(self.weight_center, a=math.sqrt(5))

        # 2. Radius initialization
        half_width = self.init_width / 2.0
        self._init_radius_raw(self.weight_radius_raw, target_radius=half_width)

        # 3. Bias initialization
        if self.bias_center is not None and self.bias_radius_raw is not None:
            fan_in = self._compute_fan_in(self.weight_shape)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
            
            nn.init.uniform_(self.bias_center, -bound, bound)
            self._init_radius_raw(self.bias_radius_raw, target_radius=half_width)

    # -----------------------------------------------------
    # String Representation (Debugging)
    # -----------------------------------------------------
    
    def extra_repr(self) -> str:
        return (f'in_dim={self.in_dim}, out_dim={self.out_dim}, '
                f'init_width={self.init_width}, bias={self.bias_center is not None}')

    # -----------------------------------------------------
    # Math Helpers
    # -----------------------------------------------------

    @staticmethod
    def _compute_fan_in(weight_shape: Tuple[int, ...]) -> int:
        if len(weight_shape) < 2: return 1
        num_input_fmaps = weight_shape[1]
        receptive_field_size = 1
        if len(weight_shape) > 2:
            receptive_field_size = math.prod(weight_shape[2:]) 
        return num_input_fmaps * receptive_field_size

    def _get_radius(self, raw_param: Tensor) -> Tensor:
        return F.softplus(raw_param)

    def _init_radius_raw(self, radius_raw: Tensor, target_radius: float) -> None:
        r = float(target_radius)
        # math.expm1(x) = exp(x) - 1, more precise for x near 0
        val = math.log(math.expm1(r)) if r > 1e-8 else math.log(r + 1e-8)
        with torch.no_grad():
            radius_raw.fill_(val)

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def get_weight_bounds(self) -> Tuple[Tensor, Tensor]:
        radius = self._get_radius(self.weight_radius_raw)
        return self.weight_center - radius, self.weight_center + radius

    def get_bias_bounds(self) -> Optional[Tuple[Tensor, Tensor]]:
        if self.bias_center is None or self.bias_radius_raw is None:
            return None
        radius = self._get_radius(self.bias_radius_raw)
        return self.bias_center - radius, self.bias_center + radius

    def get_weight_width(self) -> Tensor:
        return 2.0 * self._get_radius(self.weight_radius_raw)

    def width_loss(self, p: float = 2.0) -> Tensor:
        """
        Computes width regularization using torch.linalg.vector_norm.
        Equivalent to ||width||_p
        """
        w_width = self.get_weight_width()
        loss = torch.linalg.vector_norm(w_width, ord=p)

        if self.bias_radius_raw is not None:
            b_width = 2.0 * self._get_radius(self.bias_radius_raw)
            loss = loss + torch.linalg.vector_norm(b_width, ord=p)

        return loss

    @abstractmethod
    def forward(self, x: Tensor) -> Interval:
        raise NotImplementedError