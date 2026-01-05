from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from hgs_problem import VRPSDInstance, Solution


@dataclass
class ScenarioManager:
    """Simple Poisson scenario generator for VRPSD."""
    lam: np.ndarray
    seed: int

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def sample(self, n_scenarios: int) -> np.ndarray:
        """
        Returns int array of shape (S, n), demands for each scenario.
        Only customers 1..n-1 are Poisson; depot 0 is always 0.
        """
        n = self.lam.shape[0]
        D = self.rng.poisson(self.lam[1:], size=(n_scenarios, n - 1))
        Z = np.zeros((n_scenarios, n), dtype=int)
        Z[:, 1:] = D
        return Z


def recourse_cost_one_scenario(
    instance: VRPSDInstance,
    routes: Solution,
    demands_one: np.ndarray,
    dist_matrix: np.ndarray | None = None,
) -> float:
    """
    Execute routes under one demand realization.

    Policy: follow planned route; if capacity would be exceeded at customer c,
    go back to depot and then to c, then continue.
    """
    if dist_matrix is None:
        dist_matrix = instance.distance_matrix()
    Q = float(instance.Q)
    total = 0.0
    for r in routes:
        load = 0.0
        prev = 0
        for c in r:
            total += dist_matrix[prev, c]
            dem = float(demands_one[c])
            if load + dem > Q + 1e-9:
                total += dist_matrix[c, 0] + dist_matrix[0, c]
                load = 0.0
            load += dem
            prev = c
        total += dist_matrix[prev, 0]
    return float(total)


def saa_mean_var_std(
    instance: VRPSDInstance,
    routes: Solution,
    scenarios: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Evaluate routes on all scenarios, return (mean, variance, std).
    """
    dist = instance.distance_matrix()
    costs = np.empty(scenarios.shape[0], dtype=float)
    for i, d in enumerate(scenarios):
        costs[i] = recourse_cost_one_scenario(instance, routes, d, dist)
    mean = float(costs.mean())
    var = float(costs.var(ddof=0))
    std = float(costs.std(ddof=0))
    return mean, var, std
