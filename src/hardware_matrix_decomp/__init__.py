"""Hardware Matrix Decomposition — experimental framework for mapping neural
network weight matrices onto weight-stationary hardware tiles via SVD-based
low-rank factorisation.
"""

from .decomp import SVDDecomposition
from .error_model import GaussianErrorModel
from .network import (
    DecomposedConv2d,
    DecomposedLinear,
    DecomposedMLP,
    DecomposedNetwork,
    PassthroughModule,
    unfold_input,
)
from .pipeline import LayerPipeline
from .simulator import SimulationResult, run_simulation
from .tile import MAX_TILE_DIM, HardwareTile
from .tile_grid import TileGrid

__all__ = [
    "DecomposedConv2d",
    "DecomposedLinear",
    "DecomposedMLP",
    "DecomposedNetwork",
    "GaussianErrorModel",
    "HardwareTile",
    "LayerPipeline",
    "MAX_TILE_DIM",
    "PassthroughModule",
    "SimulationResult",
    "SVDDecomposition",
    "TileGrid",
    "run_simulation",
    "unfold_input",
]
