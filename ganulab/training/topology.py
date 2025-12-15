from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import copy
import torch
import torch.nn as nn

# Imports internos
from ganulab.modeling.neuralmodel import NeuralModel
from ganulab.training.seeding import SeedManager

# ==============================================================================
# 1. El Átomo: TopologyNode
# ==============================================================================

@dataclass
class TopologyNode:
    """
    Representa una unidad única de entrenamiento dentro de una topología.
    Puede ser un modelo singular, o un miembro específico de un ensemble/cluster.
    """
    model: NeuralModel
    id: str                 # Identificador único (ej: "row0_col1_member3")
    coords: Tuple[int, ...] # Coordenadas (row, col, member_idx)
    
    # Configuración de Entrenamiento específica para este nodo
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    
    # Configuración de Datos (para Bootstrap / Shuffling)
    data_seed: int = 42


# ==============================================================================
# 2. Clase Base: Topology
# ==============================================================================

class Topology(ABC):
    """
    Define la estructura de modelos a entrenar.
    Actúa como un 'Iterable' que entrega TopologyNodes al Engine.
    """
    def __init__(self, template_model: NeuralModel):
        self.template_model = template_model
        self.nodes: List[TopologyNode] = []
        self._is_built = False

    @abstractmethod
    def build(self) -> None:
        """Construye los nodos (clona modelos, asigna configs)."""
        pass

    def __iter__(self) -> Iterator[TopologyNode]:
        if not self._is_built:
            self.build()
            self._is_built = True
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    # ------------------------------------------------------------------
    # Helpers para clonar y reinicializar
    # ------------------------------------------------------------------
    def _clone_model(self, reset_weights: bool = False) -> NeuralModel:
        """
        Crea una copia profunda del modelo plantilla.
        """
        # 1. Deepcopy de la estructura completa
        clone = copy.deepcopy(self.template_model)
        
        # 2. Reconstruir optimizador (para desligarlo del original)
        clone.build_optimizer()
        
        # 3. Reinicializar pesos si se solicita (para Ensembles)
        if reset_weights:
            self._reset_weights(clone.module)
            
        return clone

    def _reset_weights(self, module: nn.Module) -> None:
        """
        Intenta llamar a reset_parameters() recursivamente.
        """
        # Si el módulo tiene reset_parameters, lo llamamos
        if hasattr(module, "reset_parameters") and callable(module.reset_parameters):
            module.reset_parameters()
        
        # Si es un contenedor (Sequential, ModuleList, NeuralBlock), iteramos hijos
        for child in module.children():
            self._reset_weights(child)


# ==============================================================================
# 3. Topología Singular (1 Modelo)
# ==============================================================================

class SingleTopology(Topology):
    """
    Entrena una única instancia del modelo.
    """
    def __init__(
        self, 
        template_model: NeuralModel, 
        epochs: int = 10, 
        batch_size: int = 32,
        seed: int = 42
    ):
        super().__init__(template_model)
        self.defaults = {"epochs": epochs, "batch_size": batch_size}
        self.seed = seed

    def build(self) -> None:
        # No clonamos ni reseteamos (usamos el template directo o una copia exacta)
        # Usamos copia para no alterar el template original por seguridad
        model = self._clone_model(reset_weights=False)
        
        node = TopologyNode(
            model=model,
            id="single",
            coords=(0, 0),
            hyperparams=self.defaults,
            data_seed=self.seed
        )
        self.nodes = [node]


# ==============================================================================
# 4. Topología Ensemble (N Modelos)
# ==============================================================================

class EnsembleTopology(Topology):
    """
    Entrena N copias del modelo con distintas inicializaciones.
    Soporta Bootstrap (cada modelo ve datos mezclados distinto).
    """
    def __init__(
        self,
        template_model: NeuralModel,
        n_members: int = 5,
        bootstrap: bool = True,
        # Hiperparámetros (fijos para todo el ensemble)
        epochs: int = 10,
        batch_size: int = 32,
        base_seed: int = 42
    ):
        super().__init__(template_model)
        self.n_members = n_members
        self.bootstrap = bootstrap
        self.defaults = {"epochs": epochs, "batch_size": batch_size}
        self.base_seed = base_seed

    def build(self) -> None:
        self.nodes = []
        for i in range(self.n_members):
            # 1. Clonar y Reinicializar Pesos (Clave para Ensembles)
            # Para que sean modelos distintos, deben empezar en lugares distintos
            model = self._clone_model(reset_weights=True)
            
            # 2. Configurar Semilla de Datos
            # Si bootstrap=True, variamos la semilla por miembro
            # Si bootstrap=False, todos ven el mismo orden de datos
            d_seed = self.base_seed + i if self.bootstrap else self.base_seed
            
            node = TopologyNode(
                model=model,
                id=f"ens_m{i}",
                coords=(0, i),
                hyperparams=self.defaults,
                data_seed=d_seed
            )
            self.nodes.append(node)


# ==============================================================================
# 5. Topología Cluster (Matriz de Ensembles)
# ==============================================================================

class ClusterTopology(Topology):
    """
    Una matriz de experimentos.
    
    Filas (rows): Configuraciones estructurales distintas (o simplemente semillas distintas).
                  Representan 'Grupos de Ensembles'.
    Columnas (cols): Configuraciones de Hiperparámetros (Grid Search).
    Profundidad: Miembros del ensemble en esa celda.
    
    Ejemplo:
      - Row 0: Ensemble A
      - Row 1: Ensemble B
      - Col 0: Epochs=10, Batch=32
      - Col 1: Epochs=20, Batch=64
    """
    def __init__(
        self,
        template_model: NeuralModel,
        n_rows: int = 1,          # Cuántos ensembles distintos por columna
        n_members_per_ens: int = 3, # Tamaño de cada ensemble
        configs: List[Dict[str, Any]] = None, # Lista de configs para las columnas
        bootstrap: bool = True,
        base_seed: int = 42
    ):
        super().__init__(template_model)
        self.n_rows = n_rows
        self.n_members = n_members_per_ens
        self.bootstrap = bootstrap
        self.base_seed = base_seed
        
        # Si no dan configs, usamos una default
        self.configs = configs or [{"epochs": 10, "batch_size": 32}]

    def build(self) -> None:
        self.nodes = []
        
        # Iterar Columnas (Configuraciones)
        for col_idx, config in enumerate(self.configs):
            
            # Iterar Filas (Grupos de Ensembles)
            for row_idx in range(self.n_rows):
                
                # Iterar Miembros del Ensemble
                for m_idx in range(self.n_members):
                    
                    # Clonar y Resetear
                    model = self._clone_model(reset_weights=True)
                    
                    # Semilla única calculada por coordenadas
                    # Formula para evitar colisiones de semillas
                    seed_offset = (col_idx * 1000) + (row_idx * 100) + m_idx
                    d_seed = self.base_seed + seed_offset if self.bootstrap else self.base_seed
                    
                    node = TopologyNode(
                        model=model,
                        id=f"r{row_idx}_c{col_idx}_m{m_idx}",
                        coords=(row_idx, col_idx, m_idx),
                        hyperparams=config, # La config de esta columna
                        data_seed=d_seed
                    )
                    self.nodes.append(node)