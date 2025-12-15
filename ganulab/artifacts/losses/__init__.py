# ganulab/artifacts/losses/__init__.py

import importlib
import pkgutil

from ganulab.utils.catalog import Catalog

# 1. Instanciamos el Catálogo
lflib = Catalog("lflib", description="Librería de funciones de pérdida")

# 2. CREAR Y CAPTURAR (El truco está aquí)
# Al asignar el resultado a una variable, creamos un "alias" local.
lossterm = lflib.create("lossterm", "Loss functions that can be combined in other loss functions")
penalty = lflib.create("penalty", "Penalizations for loss functions")
lossfunc = lflib.create("lossfunc", "Loss functions that can be used directly in training")

# 3. Carga Dinámica
# Importante: Definir las variables (paso 2) ANTES de importar los submódulos
# para evitar errores de importación circular si ellos las necesitan.
for info in pkgutil.walk_packages(__path__, prefix=__name__ + "."):
    importlib.import_module(info.name)

# 4. Exponemos todo
# Exponemos 'lflib' (para el usuario) y las categorías (para importar internamente si hiciera falta)
__all__ = ["lflib", "lossterm", "penalty", "lossfunc"]