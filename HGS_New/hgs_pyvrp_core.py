from __future__ import annotations

from typing import List

import numpy as np
from pyvrp import Model, solve
from pyvrp.stop import MaxRuntime

from hgs_problem import VRPSDInstance, Solution


class HGSPyVRPResult:
    """
    Thin wrapper around PyVRP's result we return to the runner script.
    """
    def __init__(self, routes: Solution, cost: float, raw_result) -> None:
        self.routes = routes
        self.cost = float(cost)
        self.raw_result = raw_result  # pyvrp.Result


def solve_vrpsd_with_hgs(
    instance: VRPSDInstance,
    time_limit_sec: float = 10.0,
    seed: int = 0,
    verbose: bool = False,
) -> HGSPyVRPResult:
    """
    Solve deterministic CVRP using PyVRP / HGS.

    Distance matrix from instance; stochastic part only used in SAA.
    """
    dist = instance.distance_matrix()
    n, m = dist.shape
    if n != m:
        raise ValueError(f"distance matrix must be square, got {dist.shape}")

    Q = int(round(instance.Q))
    if Q <= 0:
        raise ValueError("Vehicle capacity Q must be positive.")

    # 用期望需求（四舍五入成整数）作为 deterministic CVRP 的需求
    demands = instance.lam.astype(float)
    demands_int = np.rint(demands).astype(int)

    model = Model()

    # 单一 depot
    depot = model.add_depot(x=0.0, y=0.0)

    # PyVRP 新版接口：capacity, num_available；不再有 depot 参数
    model.add_vehicle_type(
        capacity=Q,
        num_available=int(instance.K) if instance.K is not None else n - 1,
    )

    # clients: 1..n-1
    for idx in range(1, n):
        model.add_client(
            x=float(idx),
            y=0.0,
            delivery=int(demands_int[idx]),
        )

    locations = model.locations
    dist_int = np.rint(dist).astype(int)
    for i, frm in enumerate(locations):
        for j, to in enumerate(locations):
            if i == j:
                continue
            model.add_edge(frm, to, distance=int(dist_int[i, j]))

    data = model.data()
    res = solve(
        data=data,
        stop=MaxRuntime(float(time_limit_sec)),
        seed=int(seed),
        display=verbose,
    )

    best = res.best
    routes: List[List[int]] = []
    for route in best.routes():
        visits = list(route.visits())
        if visits:
            routes.append(visits)

    cost = float(res.cost())
    return HGSPyVRPResult(routes=routes, cost=cost, raw_result=res)
