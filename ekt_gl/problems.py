"""Routing task definitions with a permutation genotype for every task type."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

Route = tuple[int, ...]


def _normalise_coordinates(coordinates: np.ndarray, mode: str = "per_axis") -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 3:
        raise ValueError("coordinates must have shape [node_count >= 3, 2]")
    if mode == "global":
        low, high = coordinates.min(), coordinates.max()
    elif mode == "per_axis":
        low, high = coordinates.min(axis=0), coordinates.max(axis=0)
    else:
        raise ValueError(f"unknown coordinate normalization: {mode}")
    scale = np.maximum(high - low, 1e-8)
    return (coordinates - low) / scale


@dataclass(slots=True)
class RoutingTask(ABC):
    """A route task. ``route`` is always a permutation of ``route_nodes``."""

    name: str
    coordinates: np.ndarray
    _distance: np.ndarray = field(init=False, repr=False)
    _normalization: ClassVar[str] = "per_axis"

    def __post_init__(self) -> None:
        self.coordinates = _normalise_coordinates(self.coordinates, self._normalization)
        delta = self.coordinates[:, None, :] - self.coordinates[None, :, :]
        self._distance = np.linalg.norm(delta, axis=-1)

    @property
    def node_count(self) -> int:
        return len(self.coordinates)

    @property
    @abstractmethod
    def route_nodes(self) -> tuple[int, ...]:
        """Nodes represented by the evolutionary permutation."""

    def validate_route(self, route: Route) -> None:
        if len(route) != len(self.route_nodes) or set(route) != set(self.route_nodes):
            raise ValueError(f"invalid route for task {self.name!r}")

    @abstractmethod
    def evaluate(self, route: Route) -> float:
        """Return the route cost, including all depot returns when applicable."""

    @abstractmethod
    def edge_matrix(self, route: Route) -> np.ndarray:
        """Return directed binary edges for a feasible route."""


@dataclass(slots=True)
class TSPTask(RoutingTask):
    """Symmetric Euclidean TSP with shape-preserving scalar normalization.

    A single coordinate min/max reproduces the scale of Table IX's TSP LKH
    values. Per-axis scaling would change the underlying Euclidean problem.
    """

    _normalization: ClassVar[str] = "global"

    @property
    def route_nodes(self) -> tuple[int, ...]:
        return tuple(range(self.node_count))

    def evaluate(self, route: Route) -> float:
        self.validate_route(route)
        tour = np.asarray(route, dtype=int)
        return float(self._distance[tour, np.roll(tour, -1)].sum())

    def edge_matrix(self, route: Route) -> np.ndarray:
        self.validate_route(route)
        matrix = np.zeros((self.node_count, self.node_count), dtype=np.float32)
        for source, target in zip(route, route[1:] + route[:1]):
            matrix[source, target] = matrix[target, source] = 1.0
        return matrix


@dataclass(slots=True)
class CVRPTask(RoutingTask):
    """Euclidean CVRP task; node 0 is the depot and is excluded from genotype."""

    demands: np.ndarray
    capacity: float

    def __post_init__(self) -> None:
        super(CVRPTask, self).__post_init__()
        self.demands = np.asarray(self.demands, dtype=np.float32)
        if self.demands.shape != (self.node_count,):
            raise ValueError("demands must have one value per coordinate")
        if self.demands[0] != 0 or np.any(self.demands[1:] <= 0):
            raise ValueError("depot demand must be 0 and customer demands must be positive")
        if self.capacity <= 0 or np.any(self.demands[1:] > self.capacity):
            raise ValueError("capacity must cover every individual customer demand")

    @property
    def route_nodes(self) -> tuple[int, ...]:
        return tuple(range(1, self.node_count))

    def split_routes(self, route: Route) -> list[Route]:
        """Greedily split a customer ordering into capacity-feasible depot tours."""
        self.validate_route(route)
        tours: list[Route] = []
        current: list[int] = []
        load = 0.0
        for customer in route:
            demand = float(self.demands[customer])
            if current and load + demand > self.capacity + 1e-7:
                tours.append(tuple(current))
                current, load = [], 0.0
            current.append(customer)
            load += demand
        if current:
            tours.append(tuple(current))
        return tours

    def evaluate(self, route: Route) -> float:
        total = 0.0
        for tour in self.split_routes(route):
            nodes = (0,) + tour + (0,)
            total += float(self._distance[np.asarray(nodes[:-1]), np.asarray(nodes[1:])].sum())
        return total

    def edge_matrix(self, route: Route) -> np.ndarray:
        matrix = np.zeros((self.node_count, self.node_count), dtype=np.float32)
        for tour in self.split_routes(route):
            nodes = (0,) + tour + (0,)
            for source, target in zip(nodes[:-1], nodes[1:]):
                matrix[source, target] = 1.0
        return matrix
