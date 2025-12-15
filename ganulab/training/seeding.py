import random
import os
import numpy as np
import torch

class SeedManager:
    """
    Gestor centralizado de reproductibilidad.
    """
    @staticmethod
    def set_seed(seed: int, deterministic: bool = False) -> None:
        """
        Fija la semilla en todos los generadores de números aleatorios relevantes.
        
        :param seed: Entero con la semilla.
        :param deterministic: Si True, fuerza operaciones deterministas en CuDNN 
                              (puede ser más lento).
        """
        # 1. Python nativo
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # 2. Numpy
        np.random.seed(seed)
        
        # 3. PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # Si hay multi-GPU

        # 4. Determinismo (Opcional, útil para debug estricto)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # torch.use_deterministic_algorithms(True) # Cuidado: algunas ops no lo soportan
            print(f"[SeedManager] Modo determinista activado con seed={seed}.")
        else:
            print(f"[SeedManager] Semilla fijada en {seed}.")