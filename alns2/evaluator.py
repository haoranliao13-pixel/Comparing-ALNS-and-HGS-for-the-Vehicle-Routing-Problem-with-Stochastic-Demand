
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np

try:
    from .problem import VRPSDInstance, Solution, normalize_routes
except ImportError:
    from problem import VRPSDInstance, Solution, normalize_routes

@dataclass
class ScenarioManager:
    lam: np.ndarray
    seed: int
    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
    def sample(self, n_scenarios: int) -> np.ndarray:
        n = self.lam.shape[0]
        D = self.rng.poisson(self.lam[1:], size=(n_scenarios, n - 1))
        Z = np.zeros((n_scenarios, n), dtype=int)
        Z[:, 1:] = D
        return Z

def recourse_cost_one_scenario(instance: VRPSDInstance, routes: Solution, demands_one: np.ndarray, dist_matrix: Optional[np.ndarray] = None) -> float:
    if dist_matrix is None:
        dist_matrix = instance.distance_matrix()
    Q = instance.Q
    total = 0.0
    for r in routes:
        load = 0.0; prev = 0
        for c in r:
            total += dist_matrix[prev, c]
            dem = float(demands_one[c])
            if load + dem > Q + 1e-9:
                total += dist_matrix[c, 0] + dist_matrix[0, c]
                load = 0.0
            load += dem
            prev = c
        total += dist_matrix[prev, 0]
    return total

def recourse_cost_and_restocks_one_scenario(
    instance: VRPSDInstance,
    routes: Solution,
    demands_one: np.ndarray,
    dist_matrix: Optional[np.ndarray] = None,
) -> Tuple[float, int]:
    """Return (total_cost, restock_count) for one demand scenario.

    A restock event occurs when the vehicle arrives at customer c and the remaining
    load is insufficient (load + demand > Q), triggering a depot round-trip c->0->c.
    """
    if dist_matrix is None:
        dist_matrix = instance.distance_matrix()
    Q = instance.Q
    total = 0.0
    restocks = 0
    for r in routes:
        load = 0.0
        prev = 0
        for c in r:
            total += dist_matrix[prev, c]
            dem = float(demands_one[c])
            if load + dem > Q + 1e-9:
                total += dist_matrix[c, 0] + dist_matrix[0, c]
                restocks += 1
                load = 0.0
            load += dem
            prev = c
        total += dist_matrix[prev, 0]
    return total, restocks


def trips_one_scenario(
    instance: VRPSDInstance,
    routes: Solution,
    demands_one: np.ndarray,
    dist_matrix: Optional[np.ndarray] = None,
) -> int:
    """Number of depot departures (planned routes + restocks) for one scenario."""
    routes_norm, _ = normalize_routes(routes)
    if dist_matrix is None:
        dist_matrix = instance.distance_matrix()
    _, restocks = recourse_cost_and_restocks_one_scenario(instance, routes_norm, demands_one, dist_matrix)
    return len(routes_norm) + restocks

@dataclass
class SAACache:
    cost_small: Dict[Tuple[Tuple[int, ...], ...], float]
    cost_large: Dict[Tuple[Tuple[int, ...], ...], float]
    def __init__(self):
        self.cost_small = {}; self.cost_large = {}

def expected_cost_SAA(instance: VRPSDInstance, routes: Solution, scenarios: np.ndarray) -> float:
    routes_norm, _ = normalize_routes(routes)
    dist = instance.distance_matrix()
    tot = 0.0
    for d in scenarios:
        tot += recourse_cost_one_scenario(instance, routes_norm, d, dist)
    return tot / scenarios.shape[0]

def evaluate_routes(instance: VRPSDInstance, routes: Solution, cache: Optional[SAACache], scen_small: np.ndarray, scen_large: Optional[np.ndarray]):
    _, key = normalize_routes(routes)
    if cache is not None and key in cache.cost_small:
        small = cache.cost_small[key]
    else:
        small = expected_cost_SAA(instance, routes, scen_small)
        if cache is not None:
            cache.cost_small[key] = small
    large = None
    if scen_large is not None:
        if cache is not None and key in cache.cost_large:
            large = cache.cost_large[key]
        else:
            large = expected_cost_SAA(instance, routes, scen_large)
            if cache is not None:
                cache.cost_large[key] = large
    return small, large
