from __future__ import annotations

import os
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt

from hgs_problem import build_instance_from_csv, solution_cost_deterministic
from hgs_saa import ScenarioManager, saa_mean_var_std
from hgs_pyvrp_core import solve_vrpsd_with_hgs


@dataclass
class HGSVRPSDConfig:
    # dataset
    nodes_csv: str
    demand_csv: str

    # VRPSD params
    Q: str
    K: str

    # SAA params
    small_S: int = 80
    big_S: int = 300
    seed_small: int = 12345
    seed_big: int = 54321

    # HGS / PyVRP params
    time_limit_sec: float = 10.0
    hgs_seed: int = 0
    verbose: bool = False


def run_experiment(cfg: HGSVRPSDConfig) -> None:
    # ---------- 1) load instance ----------
    inst = build_instance_from_csv(cfg.nodes_csv, cfg.demand_csv, Q=cfg.Q, K=cfg.K)

    # ---------- 2) deterministic CVRP with HGS ----------
    t0 = time.time()
    hgs_res = solve_vrpsd_with_hgs(
        inst,
        time_limit_sec=cfg.time_limit_sec,
        seed=cfg.hgs_seed,
        verbose=cfg.verbose,
    )
    t1 = time.time()
    solve_time = t1 - t0

    routes = hgs_res.routes
    det_cost = solution_cost_deterministic(inst, routes)

    # ---------- 3) SAA evaluation (small + big) ----------
    scen_small = ScenarioManager(inst.lam, cfg.seed_small).sample(cfg.small_S)
    scen_big = ScenarioManager(inst.lam, cfg.seed_big).sample(cfg.big_S)

    mean_small, var_small, std_small = saa_mean_var_std(inst, routes, scen_small)
    mean_big, var_big, std_big = saa_mean_var_std(inst, routes, scen_big)

    # ---------- 4) console report ----------
    print("===== HGS (PyVRP) VRPSD evaluation =====")
    print(f"Deterministic HGS distance: {det_cost:.3f}")
    print(f"Number of routes: {len(routes)}")
    print(f"Solve time (PyVRP): {solve_time:.3f} s\n")

    print(f"[Small-SAA] S={cfg.small_S}, seed={cfg.seed_small}")
    print(f"  mean = {mean_small:.4f}, var = {var_small:.4f}, std = {std_small:.4f}")
    print(f"[Big-SAA]   S={cfg.big_S}, seed={cfg.seed_big}")
    print(f"  mean = {mean_big:.4f}, var = {var_big:.4f}, std = {std_big:.4f}")

    # ---------- 5) plots ----------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # 5.1 routes plot
    fig_routes, ax_routes = plt.subplots(figsize=(6, 6))
    coords = inst.coords
    ax_routes.scatter(coords[1:, 0], coords[1:, 1], s=10, label="Customers")
    ax_routes.scatter(coords[0, 0], coords[0, 1], s=40, label="Depot")
    for r in routes:
        seq = [0] + r + [0]
        ax_routes.plot(coords[seq, 0], coords[seq, 1], linewidth=1)
    ax_routes.set_title("HGS routes (deterministic CVRP)")
    ax_routes.legend()
    ax_routes.set_aspect("equal", "box")
    fig_routes.tight_layout()
    routes_png = os.path.join(out_dir, "hgs_vrpsd_routes.png")
    fig_routes.savefig(routes_png, dpi=300)
    print("Saved route plot to:", routes_png)

    # 5.2 convergence plot from PyVRP result
    try:
        from pyvrp.plotting import plot_objectives

        fig_conv, ax_conv = plt.subplots(figsize=(6, 4))
        plot_objectives(hgs_res.raw_result, ax=ax_conv)
        ax_conv.set_title("HGS (PyVRP) convergence")
        ax_conv.set_xlabel("Iteration")
        ax_conv.set_ylabel("Objective (total distance)")
        fig_conv.tight_layout()
        conv_png = os.path.join(out_dir, "hgs_vrpsd_convergence.png")
        fig_conv.savefig(conv_png, dpi=300)
        print("Saved convergence plot to:", conv_png)
    except Exception as e:
        print("[WARN] Could not plot convergence curve from PyVRP result:", e)

    plt.show()


if __name__ == "__main__":
    # 推断项目根目录：.../Comparing-ALNS-and-HGS.../
    this_dir = os.path.dirname(os.path.abspath(__file__))
    hsg_vrpsd_dir = os.path.dirname(this_dir)
    root_dir = os.path.dirname(hsg_vrpsd_dir)

    # 默认用 CVRPLib/csv 下的 X-n101-k25
    nodes_csv = os.path.join(root_dir, "CVRPLib", "csv", "nodes_X-n101-k25.csv")
    demand_csv = os.path.join(root_dir, "CVRPLib", "csv", "demand_X-n101-k25.csv")

    cfg = HGSVRPSDConfig(
        nodes_csv=nodes_csv,
        demand_csv=demand_csv,
        Q=220.0,
        K=90,
        time_limit_sec=15.0,
        hgs_seed=0,
        verbose=False,
    )
    run_experiment(cfg)
