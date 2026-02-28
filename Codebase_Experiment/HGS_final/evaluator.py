from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from problem import VRPSDInstance, Solution


@dataclass
class ScenarioManager:
    
    lam: np.ndarray
    seed: int

    def __post_init__(self) -> None:
        self.lam = np.asarray(self.lam, dtype=float)
        self.rng = np.random.default_rng(int(self.seed))

    def sample(self, S: int) -> np.ndarray:
        n = int(self.lam.shape[0])
        S = int(S)
        if S <= 0:
            raise ValueError("S must be positive.")
        if n <= 0:
            raise ValueError("lam must be non-empty.")

        # sample customers only, keep depot demand=0
        cust = self.rng.poisson(self.lam[1:], size=(S, n - 1))
        out = np.zeros((S, n), dtype=int)
        out[:, 1:] = cust
        return out


def recourse_cost_one_scenario(
    instance: VRPSDInstance,
    routes: Solution,
    demands_one: np.ndarray,
    dist_matrix: Optional[np.ndarray] = None,
) -> float:
    
    if dist_matrix is None:
        dist_matrix = instance.distance_matrix()

    Q = float(instance.Q)
    demands_one = np.asarray(demands_one)

    total = 0.0
    for r in routes:
        load = 0.0
        prev = 0

        for c in r:
            total += float(dist_matrix[prev, c])

            dem = float(demands_one[c])
            if load + dem > Q + 1e-9:
                total += float(dist_matrix[c, 0]) + float(dist_matrix[0, c])
                load = 0.0

            load += dem
            prev = c

        total += float(dist_matrix[prev, 0])

    return float(total)


def saa_mean_var_std(
    instance: VRPSDInstance,
    routes: Solution,
    scenarios: np.ndarray,
) -> Tuple[float, float, float]:
    
    scenarios = np.asarray(scenarios)
    if scenarios.ndim != 2:
        raise ValueError("scenarios must be a 2D array of shape (S, n).")

    D = instance.distance_matrix()
    S = scenarios.shape[0]
    costs = np.empty(S, dtype=float)

    for i in range(S):
        costs[i] = recourse_cost_one_scenario(instance, routes, scenarios[i], D)

    mean = float(costs.mean())
    var = float(costs.var(ddof=0))
    std = float(costs.std(ddof=0))
    return mean, var, std

