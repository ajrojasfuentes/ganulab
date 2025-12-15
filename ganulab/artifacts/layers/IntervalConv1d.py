# ganulab/artifacts/layers/IntervalConv1d.py

from typing import Tuple, Union
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _single

from ganulab.modeling.neurallayer import IntervalLayer, split_pos_neg, Interval
from . import interval

@interval(name="Conv1d")
class IntervalConv1d(IntervalLayer):
    """
    Capa convolucional 1D intervalar.
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
        init_width: float = 1e-2,
    ):
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.dilation = _single(dilation)
        self.groups = groups

        weight_shape = (out_channels, in_channels // groups, self.kernel_size[0])

        super().__init__(
            in_dim=in_channels,
            out_dim=out_channels,
            weight_shape=weight_shape,
            bias=bias,
            init_width=init_width,
        )

    def forward(self, x: torch.Tensor) -> Interval:
        # JIT function
        x_pos, x_neg = split_pos_neg(x)
        
        W_lower, W_upper = self.get_weight_bounds()

        # Helper para conv1d con params fijos
        def conv(input, weight):
            return F.conv1d(
                input, weight, None, 
                self.stride, self.padding, self.dilation, self.groups
            )

        # Propagación intervalar
        y_lower = conv(x_pos, W_lower) + conv(x_neg, W_upper)
        y_upper = conv(x_pos, W_upper) + conv(x_neg, W_lower)

        b_bounds = self.get_bias_bounds()
        if b_bounds is not None:
            # Broadcasting: [C_out] -> [1, C_out, 1]
            b_l = b_bounds[0].view(1, -1, 1)
            b_u = b_bounds[1].view(1, -1, 1)
            y_lower += b_l
            y_upper += b_u

        return y_lower, y_upper

    def extra_repr(self) -> str:
        s = super().extra_repr()
        return s + (f', kernel_size={self.kernel_size}, stride={self.stride}, '
                    f'padding={self.padding}, dilation={self.dilation}, groups={self.groups}')