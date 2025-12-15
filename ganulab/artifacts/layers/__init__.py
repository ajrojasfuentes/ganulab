# ganulab/artifacts/layers/__init__.py

import torch.nn as nn
import pkgutil
import importlib
from ganulab.utils.catalog import Catalog

# 1. Crear el Catálogo
nllib = Catalog("nllib", description="Librería de capas neuronales")

# 2. Crear secciones
simple   = nllib.create("simple", "Capas estándar de PyTorch")
bayesian = nllib.create("bayesian", "Capas Bayesianas (Variational Inference)")
interval = nllib.create("interval", "Capas Intervalares (IBP)")

# 3. REGISTRO DE CAPAS STANDARD
# Registramos directamente las clases de PyTorch
simple(nn.Linear, name="Linear")
simple(nn.Conv1d, name="Conv1d")
simple(nn.Conv2d, name="Conv2d")
simple(nn.Conv3d, name="Conv3d")
simple(nn.ReLU, name="ReLU")
simple(nn.LeakyReLU, name="LeakyReLU")
simple(nn.Dropout, name="Dropout")
simple(nn.BatchNorm1d, name="BatchNorm1d")
simple(nn.BatchNorm2d, name="BatchNorm2d")

# 3. Registo de capas de activación en salida
simple(nn.Sigmoid, name="Sigmoid")
simple(nn.Tanh, name="Tanh")
simple(nn.Softmax, name="Softmax")
simple(nn.Identity, name="Identity")

# 4. Carga dinámica automática
# Esto busca todos los archivos .py en esta carpeta e importa los módulos.
# Al importarse, los decoradores @bayesian y @interval se ejecutan y registran las capas.
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    if module_name == '__init__':
        continue
    try:
        importlib.import_module(f'.{module_name}', package=__name__)
    except ImportError as e:
        print(f"Warning: Could not import layer module {module_name}: {e}")

# 5. Exponer
__all__ = ["nllib", "simple", "bayesian", "interval"]