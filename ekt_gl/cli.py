"""Small JSON CLI for reproducible EKT-GL experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import EKTGLConfig
from .problems import CVRPTask, RoutingTask, TSPTask
from .solver import EKTGL


def _read_task(item: dict[str, Any]) -> RoutingTask:
    kind = item["type"].lower()
    common = {"name": item["name"], "coordinates": np.asarray(item["coordinates"], dtype=np.float32)}
    if kind == "tsp":
        return TSPTask(**common)
    if kind == "cvrp":
        return CVRPTask(
            **common,
            demands=np.asarray(item["demands"], dtype=np.float32),
            capacity=float(item["capacity"]),
        )
    raise ValueError(f"unsupported task type {item['type']!r}; use 'tsp' or 'cvrp'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EKT-GL on TSP/CVRP tasks defined in JSON.")
    parser.add_argument("input", type=Path, help="JSON file with a 'tasks' array")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--transfer-interval", type=int, default=10)
    arguments = parser.parse_args()

    payload = json.loads(arguments.input.read_text(encoding="utf-8"))
    tasks = [_read_task(item) for item in payload["tasks"]]
    config = EKTGLConfig(
        generations=arguments.generations,
        population_size=arguments.population_size,
        transfer_interval=arguments.transfer_interval,
    )
    result = EKTGL(config).solve(tasks, seed=arguments.seed)
    print(json.dumps({name: {"cost": value.cost, "route": value.route} for name, value in result.best.items()}, indent=2))


if __name__ == "__main__":
    main()
