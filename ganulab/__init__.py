# ganulab/__init__.py

# ==========================
# 1) Utilities
# ==========================

from . import utils
from . import data

# ==========================
# 2) Artifacts Library
# ==========================

from .artifacts import lflib  # noqa: F401
from .artifacts import nllib  # noqa: F401

# ==========================
# 3) Modeling API
# ==========================

# Superclases y namespace de capas
from .modeling.neurallayer import (
    NeuralLayer,
    BayesianLayer,
    IntervalLayer,
)

from .modeling.neuralblock import (
    NeuralBlock
)

from .modeling.lossfunction import (
    LossFunction
)

from .modeling.neuralmodel import (
    LatentSampler,
    NeuralModel,
    inspect_model,
)

# ==========================
# 5) Versión del framework
# ==========================

__version__ = "0.1.0"

# ==========================
# 6) Exponer las funciones
# ==========================

__all__ = [
    'utils',
    'data',
    "NeuralLayer",
    "BayesianLayer",
    "IntervalLayer",
    "NeuralBlock",
    "LossFunction",
    "lflib",
    "nllib",
    "LatentSampler",
    "NeuralModel",
    "inspect_model",
]