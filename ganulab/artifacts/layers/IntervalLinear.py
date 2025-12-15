# ganulab/artifacts/layers/IntervalLinear.py

from typing import Tuple
import torch
import torch.nn.functional as F
from ganulab.modeling.neurallayer import IntervalLayer, split_pos_neg, Interval
from . import interval

@interval(name="Linear")
class IntervalLinear(IntervalLayer):
    """
    Capa lineal intervalar (IBP).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_width: float = 1e-2,
        bias: bool = True,
    ):
        super().__init__(
            in_dim=in_features,
            out_dim=out_features,
            weight_shape=(out_features, in_features),
            bias=bias,
            init_width=init_width,
        )

    def forward(self, x: torch.Tensor) -> Interval:
        # Usamos la función JIT importada
        x_pos, x_neg = split_pos_neg(x)

        # Obtenemos bounds geométricos del padre
        W_lower, W_upper = self.get_weight_bounds()
        
        # Propagación: (x+ * W-) + (x- * W+)
        y_lower = F.linear(x_pos, W_lower) + F.linear(x_neg, W_upper)
        y_upper = F.linear(x_pos, W_upper) + F.linear(x_neg, W_lower)

        b_bounds = self.get_bias_bounds()
        if b_bounds is not None:
            y_lower += b_bounds[0]
            y_upper += b_bounds[1]

        return y_lower, y_upper