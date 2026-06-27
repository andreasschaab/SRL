"""SRL — Structural Reinforcement Learning for heterogeneous-agent macroeconomic models.

A clean, pedagogical implementation of policy-gradient methods for solving
heterogeneous-agent models in general equilibrium, including with aggregate risk.
"""

__version__ = "0.1.0"

from srl import utils
from srl.spg import SPGSolver
from srl.vfi import VFISolver

__all__ = ["SPGSolver", "VFISolver", "utils", "__version__"]
