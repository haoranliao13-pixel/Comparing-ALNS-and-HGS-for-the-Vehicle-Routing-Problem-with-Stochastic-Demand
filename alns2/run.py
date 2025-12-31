import os, sys
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(__file__))
    from problem import build_instance_from_csv, k_min_lower_bound, solution_cost_deterministic
    from solver import ALNSSolveConfig, solve_vrpsd_with_alns
    from visualize import plot_routes, plot_convergence
else:
    from .problem import build_instance_from_csv, k_min_lower_bound, solution_cost_deterministic
    from .solver import ALNSSolveConfig, solve_vrpsd_with_alns
    from .visualize import plot_routes, plot_convergence

try:
    # SAA scenario generator + per-scenario recourse simulation
    from evaluator import ScenarioManager, recourse_cost_one_scenario, normalize_routes, trips_one_scenario
except ImportError:
    from .evaluator import ScenarioManager, recourse_cost_one_scenario, normalize_routes, trips_one_scenario


# ---------------- Parameters to tune ----------------
NODES_CSV  = r"C:\Users\haora\PyCharmMiscProject\CVRPLib\csv\nodes_X-n303-k21.csv"
DEMAND_CSV = r"C:\Users\haora\PyCharmMiscProject\CVRPLib\csv\demand_X-n303-k21.csv"


Q = 220.0
K = 90              # ← 与你脚本一致：不要硬约束车数量（非常关键）

CONFIG = ALNSSolveConfig(
    seed=0,                       # 你的脚本默认 seed=0
    time_limit_sec=5,         # 可选；有就设 180s
    iters=1000,                  # ≈ epochs(100) * pu(6) * 20 的同量级；给足搜索步数
    # 模拟退火：对齐“0.15×初始目标”的量级（初始通常 ~4800–5200）
    start_temp=750.0,             # ≈ 0.15 * 5000
    end_temp=1e-3,
    cool_rate=0.995,              # ← 与你脚本相同

    # SAA 样本：与脚本完全一致
    small_samples=80,             # 搜索口径 S=80
    large_samples=300,            # 复评口径 S=300

    num_starts=1,                 # 你脚本只有一个起点
    use_adaptive_selection=True,
    roulette_scores=[5.0, 2.0, 1.0, 0.2],
    roulette_decay=0.9,           # 略更贪心一点
    smooth_alpha=0.0,             # 别平滑，保持和你打印的目标一致
)


PLOT_DIR = "outputs"
os.makedirs(PLOT_DIR, exist_ok=True)

def main():
    inst = build_instance_from_csv(NODES_CSV, DEMAND_CSV, Q=Q, K=K)

    # 下界提示（只打印，不中断）
    kmin = k_min_lower_bound(inst)
    if K is not None and K < kmin:
        print("[WARN] K ({}) < lower bound ceil(sum(lam)/Q) = {}. Feasibility may be impossible."
              .format(K, kmin))

    # -------- 求解 --------
    res = solve_vrpsd_with_alns(inst, CONFIG)

    # -------- Recompute SAA mean/std on the SAME scenario sets used by the solver --------
    dist = inst.distance_matrix()
    routes_norm, _ = normalize_routes(res.routes)

    scen_small = ScenarioManager(inst.lam, seed=12345).sample(CONFIG.small_samples)
    scen_large = ScenarioManager(inst.lam, seed=54321).sample(CONFIG.large_samples)

    small_costs = np.array([recourse_cost_one_scenario(inst, routes_norm, d, dist) for d in scen_small], dtype=float)
    large_costs = np.array([recourse_cost_one_scenario(inst, routes_norm, d, dist) for d in scen_large], dtype=float)

    small_mean = float(small_costs.mean()); small_std = float(small_costs.std(ddof=1))
    large_mean = float(large_costs.mean()); large_std = float(large_costs.std(ddof=1))


    # -------- 打印结果 --------
    print(f"[Small-SAA] mean={small_mean:.4f}, std={small_std:.4f} (S={CONFIG.small_samples}, seed=12345)")
    print(f"[Big-SAA]   mean={large_mean:.4f}, std={large_std:.4f} (S={CONFIG.large_samples}, seed=54321)")
    print(f"Routes: {res.routes}")
    print(f"Number of routes: {len(res.routes)}")
    # Actual depot departures (planned routes + restocks) for ONE representative scenario
    trips0 = trips_one_scenario(inst, res.routes, scen_large[0])
    print(f"Trips (scenario 0 of big-SAA): {trips0}")

    # -------- 画图 + 保存 --------
    _ = plot_routes(inst, res.routes, fname=os.path.join(PLOT_DIR, "final_routes.png"))
    _ = plot_convergence(res.history, fname=os.path.join(PLOT_DIR, "expected_cost_convergence.png"))
    plt.tight_layout()
    plt.show()  # 两张图都画完再展示

    print(f"Saved plots to {PLOT_DIR}/final_routes.png and {PLOT_DIR}/expected_cost_convergence.png")

    # Windows 下顺手打开 PNG（可选）
    if sys.platform.startswith("win"):
        try:
            os.startfile(os.path.join(PLOT_DIR, "final_routes.png"))
            os.startfile(os.path.join(PLOT_DIR, "expected_cost_convergence.png"))
        except Exception:
            pass

    # 参考：确定性路程
    det = solution_cost_deterministic(inst, res.routes)
    print(f"Deterministic travel distance (for reference): {det:.3f}")

if __name__ == "__main__":
    main()
