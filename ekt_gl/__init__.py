"""EKT-GL: graph-learning knowledge transfer for many-task routing."""

from typing import TYPE_CHECKING

from .config import EKTGLConfig
from .problems import CVRPTask, RoutingTask, TSPTask

if TYPE_CHECKING:
    from .solver import EKTGL, EKTGLResult

__all__ = ["CVRPTask", "EKTGL", "EKTGLConfig", "EKTGLResult", "RoutingTask", "TSPTask"]


def __getattr__(name: str):
    """Defer PyTorch import until the graph-learning solver is actually used."""
    if name in {"EKTGL", "EKTGLResult"}:
        from .solver import EKTGL, EKTGLResult

        return {"EKTGL": EKTGL, "EKTGLResult": EKTGLResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
