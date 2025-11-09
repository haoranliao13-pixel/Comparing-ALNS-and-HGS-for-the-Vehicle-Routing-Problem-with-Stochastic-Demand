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

# ---------------- Parameters to tune ----------------
NODES_CSV = r"C:\Users\haora\PyCharmMiscProject\datasets\nodes_400_centered.csv"
DEMAND_CSV = r"C:\Users\haora\PyCharmMiscProject\datasets\demand_400_centered.csv"


Q = 60.0
K = 35              # ← 与你脚本一致：不要硬约束车数量（非常关键）

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

    # -------- 打印结果 --------
    print(f"Best expected cost (small sample): {res.small_cost:.3f}")
    print(f"Best expected cost (large sample):  {res.large_cost:.3f}")
    print(f"Routes: {res.routes}")

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