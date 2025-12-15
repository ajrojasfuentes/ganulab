# ganulab/artifacts/layers/BayesianLinear.py

from ganulab.modeling.neurallayer import BayesianLayer
import torch
import torch.nn.functional as F
from . import bayesian

@bayesian(name="Linear")
class BayesianLinear(BayesianLayer):
    """
    Capa lineal bayesiana (Variational Inference).
    
    Hereda de BayesianLayer:
      - Inicialización Kaiming/Uniform automática.
      - Muestreo optimizado con JIT (reparameterize).
      - Cálculo de KL divergence automático.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        prior_std: float = 1.0,  # Actualizado nombre
        learnable_prior: bool = False,
    ):
        super().__init__(
            in_dim=in_features,
            out_dim=out_features,
            weight_shape=(out_features, in_features),
            bias=bias,
            prior_std=prior_std,
            learnable_prior=learnable_prior,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # El padre ya nos da métodos sample_weight/bias optimizados
        W = self.sample_weight()
        b = self.sample_bias()
        return F.linear(x, W, b)