import os
import sys
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import time  # for measuring solve time

if __package__ is None or __package__ == "":
    here = os.path.dirname(__file__)
    if here not in sys.path:
        sys.path.append(here)
    from problem import (
        build_instance_from_csv,
        k_min_lower_bound,
        solution_cost_deterministic,
        normalize_routes,
    )
    from solver import ALNSSolveConfig, solve_vrpsd_with_alns
    from visualize import plot_routes, plot_convergence
    from evaluator import ScenarioManager, recourse_cost_one_scenario
else:  # pragma: no cover
    from .problem import (
        build_instance_from_csv,
        k_min_lower_bound,
        solution_cost_deterministic,
        normalize_routes,
    )
    from .solver import ALNSSolveConfig, solve_vrpsd_with_alns
    from .visualize import plot_routes, plot_convergence
    from .evaluator import ScenarioManager, recourse_cost_one_scenario


SMALL_SAA_SEED = 12345
BIG_SAA_SEED = 54321


def saa_mean_std_for_routes(instance, routes, n_scenarios: int, seed: int):
    scen_mgr = ScenarioManager(instance.lam, seed=seed)
    scenarios = scen_mgr.sample(n_scenarios)
    routes_norm, _ = normalize_routes(routes)
    dist = instance.distance_matrix()

    costs = []
    for dem in scenarios:
        c = recourse_cost_one_scenario(instance, routes_norm, dem, dist)
        costs.append(c)

    costs = np.asarray(costs, dtype=float)
    mean = float(costs.mean())
    std = float(costs.std(ddof=1)) if costs.size > 1 else 0.0
    return mean, std


# ---------------- Parameters to tune ----------------

# 仓库根目录：alns2/..  -> 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CVRPLib CSV 文件（如果换 instance，只改文件名就行）
NODES_CSV = os.path.join(ROOT_DIR, "CVRPLib", "csv", "nodes_X-n200-k36.csv")
DEMAND_CSV = os.path.join(ROOT_DIR, "CVRPLib", "csv", "demand_X-n200-k36.csv")

Q = 190.0
K = 30  

CONFIG = ALNSSolveConfig(
    seed=0,
    time_limit_sec=5,  # 时间上限（秒）
    iters=1000,
    start_temp=750.0,
    end_temp=1e-3,
    cool_rate=0.995,
    small_samples=80,
    large_samples=300,
    num_starts=1,
    use_adaptive_selection=True,
    roulette_scores=[5.0, 2.0, 1.0, 0.2],
    roulette_decay=0.9,
)

PLOT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def main() -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)

    # 1) 读入实例
    instance = build_instance_from_csv(NODES_CSV, DEMAND_CSV, Q=Q, K=K)

    # 2) 简单的 K 下界（只用均值需求，信息用途）
    k_lb = k_min_lower_bound(instance)
    print(f"Naive lower bound for K (ignore stochasticity): {k_lb}")

    # 3) 求解 VRPSD + 计时
    print("Solving VRPSD with ALNS...")
    t0 = time.perf_counter()
    result = solve_vrpsd_with_alns(instance, CONFIG)
    solve_time = time.perf_counter() - t0
    print(f"Solve time (ALNS): {solve_time:.3f} s")

    # -------- 基本结果 --------
    print(f"Best expected cost (small sample): {result.small_cost:.3f}")
    print(f"Best expected cost (large sample):  {result.large_cost:.3f}")

    # 与 HGS 对齐：只数非空路线
    num_routes = sum(1 for r in result.routes if r)
    print(f"Number of routes: {num_routes}")

    # 对最终策略做 SAA 统计（均值 + std）
    mean_small, std_small = saa_mean_std_for_routes(
        instance, result.routes, CONFIG.small_samples, SMALL_SAA_SEED
    )
    mean_big, std_big = saa_mean_std_for_routes(
        instance, result.routes, CONFIG.large_samples, BIG_SAA_SEED
    )

    print(
        f"[Small-SAA] mean={mean_small:.4f}, std={std_small:.4f} "
        f"(S={CONFIG.small_samples}, seed={SMALL_SAA_SEED})"
    )
    print(
        f"[Big-SAA]   mean={mean_big:.4f}, std={std_big:.4f} "
        f"(S={CONFIG.large_samples}, seed={BIG_SAA_SEED})"
    )

    print(f"Routes: {result.routes}")

    # 4) 路径图（用你统一的 academic style）
    routes_png = os.path.join(PLOT_DIR, "final_routes.png")
    plot_routes(
        instance,
        result.routes,
        fname=routes_png,
        title="Final routes (ALNS VRPSD)",
    )

    # 5) 收敛曲线：小样本 SAA 期望成本随迭代
    if result.history is not None:
        conv_png = os.path.join(PLOT_DIR, "expected_cost_convergence.png")
        plot_convergence(
            result.history,
            fname=conv_png,
            title="Expected cost (SAA small-sample) over iterations",
        )
        print(f"Saved convergence plot to {conv_png}")
    else:
        print("No convergence history stored in result.")
        conv_png = None

    print(f"Saved route plot to {routes_png}")

    # 6) 确定性均值需求下的总路程（只做参考）
    det_cost = solution_cost_deterministic(instance, result.routes)
    print(f"Deterministic travel distance (for reference): {det_cost:.3f}")


if __name__ == "__main__":
    main()
