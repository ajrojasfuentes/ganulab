# ganulab/artifacts/layers/BayesianConv1d.py

from typing import Tuple, Union
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _single

from ganulab.modeling.neurallayer import BayesianLayer
from . import bayesian

@bayesian(name="Conv1d")
class BayesianConv1d(BayesianLayer):
    """
    Capa convolucional 1D bayesiana.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        bias: bool = True,
        prior_std: float = 1.0,
        learnable_prior: bool = False,
    ):
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.dilation = _single(dilation)
        self.groups = groups

        # Shape estándar de Conv1d: [C_out, C_in/groups, K]
        weight_shape = (out_channels, in_channels // groups, self.kernel_size[0])

        super().__init__(
            in_dim=in_channels,
            out_dim=out_channels,
            weight_shape=weight_shape,
            bias=bias,
            prior_std=prior_std,
            learnable_prior=learnable_prior,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.sample_weight()
        bias = self.sample_bias()

        return F.conv1d(
            x,
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    # Solo sobreescribimos extra_repr para añadir los hyperparams de convolución
    def extra_repr(self) -> str:
        s = super().extra_repr() # Obtenemos la info base (dims, prior)
        return s + (f', kernel_size={self.kernel_size}, stride={self.stride}, '
                    f'padding={self.padding}, dilation={self.dilation}, groups={self.groups}')