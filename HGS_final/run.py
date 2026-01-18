from __future__ import annotations

import os
import sys
import time
import json

INSTANCES = ["X-n101-k25"]     # e.g., ["X-n101-k25", "X-n219-k73"]

# Override with explicit CSV paths (set BOTH, or keep BOTH as None)
NODES_CSV = None              # e.g., r"C:\...\CVRPLib\csv\nodes_X-n101-k25.csv"
DEMAND_CSV = None             # e.g., r"C:\...\CVRPLib\csv\demand_X-n101-k25.csv"

# Problem parameters
Q = 220.0                     # vehicle capacity
K = 90                        # max number of vehicles

# HGS / PyVRP parameters
TIME_LIMIT_SEC = 15.0         # solver runtime limit (seconds)
HGS_SEED = 0                  # random seed for PyVRP
VERBOSE = False               # PyVRP display output

# SAA evaluation parameters
SMALL_S = 80                  # small-sample scenario count
BIG_S = 300                   # large-sample scenario count
SEED_SMALL = 12345            # RNG seed for small scenarios
SEED_BIG = 54321              # RNG seed for big scenarios

# Output controls
OUT_DIR = "outputs"           # relative to HGS_final/ by default
DISABLE_PLOTS = False         # True -> do not save plots
SAVE_NATIVE_CONV = False      # True -> also save PyVRP native objective plot
SAVE_JSON_SUMMARY = True      # True -> write hgs_summary.json
# =============================================================================


# -----------------------------------------------------------------------------
# Make sure imports resolve to *this* folder first (avoid alns2/problem.py clash)
# -----------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from problem import build_instance_from_csv, solution_cost_deterministic  # noqa: E402
from solver import solve_vrpsd_with_hgs  # noqa: E402
from evaluator import ScenarioManager, saa_mean_var_std  # noqa: E402
from visualize import (  # noqa: E402
    plot_routes,
    plot_convergence_alns_style,
    plot_convergence_pyvrp_native,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def print_parameter_console() -> None:
    """Print parameter summary at the very top (English)."""
    print("===== Parameter Console (edit in run.py) =====")
    print(f"INSTANCES           = {INSTANCES}")
    print(f"NODES_CSV           = {NODES_CSV}")
    print(f"DEMAND_CSV          = {DEMAND_CSV}")
    print()
    print("[Problem]")
    print(f"  Q                 = {Q}")
    print(f"  K                 = {K}")
    print()
    print("[HGS / PyVRP]")
    print(f"  TIME_LIMIT_SEC    = {TIME_LIMIT_SEC}")
    print(f"  HGS_SEED          = {HGS_SEED}")
    print(f"  VERBOSE           = {VERBOSE}")
    print()
    print("[SAA Evaluation]")
    print(f"  SMALL_S           = {SMALL_S}   (seed={SEED_SMALL})")
    print(f"  BIG_S             = {BIG_S}     (seed={SEED_BIG})")
    print()
    print("[Output]")
    print(f"  OUT_DIR           = {OUT_DIR}")
    print(f"  DISABLE_PLOTS     = {DISABLE_PLOTS}")
    print(f"  SAVE_NATIVE_CONV  = {SAVE_NATIVE_CONV}")
    print(f"  SAVE_JSON_SUMMARY = {SAVE_JSON_SUMMARY}")
    print("==============================================\n")


def find_project_root(start_dir: str) -> str:
    """
    Walk up to find a directory that contains 'CVRPLib'.
    This makes paths robust across machines / different folder locations.
    """
    cur = os.path.abspath(start_dir)
    for _ in range(10):
        if os.path.isdir(os.path.join(cur, "CVRPLib")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(start_dir)


def resolve_csv_paths(project_root: str, instance_name: str) -> tuple[str, str]:
    """
    If NODES_CSV/DEMAND_CSV are provided, use them.
    Else use CVRPLib/csv/nodes_<instance>.csv and demand_<instance>.csv.
    """
    if (NODES_CSV is None) ^ (DEMAND_CSV is None):
        raise ValueError("You must set BOTH NODES_CSV and DEMAND_CSV, or set BOTH to None.")

    if NODES_CSV is not None and DEMAND_CSV is not None:
        return NODES_CSV, DEMAND_CSV

    csv_dir = os.path.join(project_root, "CVRPLib", "csv")
    nodes = os.path.join(csv_dir, f"nodes_{instance_name}.csv")
    demand = os.path.join(csv_dir, f"demand_{instance_name}.csv")
    return nodes, demand


def ensure_out_dir(base_out_dir: str, instance_name: str) -> str:
    """Create an instance-specific output folder to avoid overwriting files."""
    out_dir = base_out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(THIS_DIR, out_dir)
    out_dir = os.path.join(out_dir, instance_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# -----------------------------------------------------------------------------
# Core routine
# -----------------------------------------------------------------------------
def run_one_instance(instance_name: str) -> None:
    project_root = find_project_root(THIS_DIR)
    nodes_csv, demand_csv = resolve_csv_paths(project_root, instance_name)
    out_dir = ensure_out_dir(OUT_DIR, instance_name)

    print("===== Environment / Paths =====")
    print(f"Instance     : {instance_name}")
    print(f"Project root : {project_root}")
    print(f"Nodes CSV    : {nodes_csv}")
    print(f"Demand CSV   : {demand_csv}")
    print(f"Output dir   : {out_dir}")
    print()

    if not os.path.isfile(nodes_csv):
        raise FileNotFoundError(f"Nodes CSV not found: {nodes_csv}")
    if not os.path.isfile(demand_csv):
        raise FileNotFoundError(f"Demand CSV not found: {demand_csv}")

    # 1) Build instance
    inst = build_instance_from_csv(nodes_csv, demand_csv, Q=float(Q), K=int(K))

    # 2) Solve deterministic CVRP approximation via HGS (PyVRP)
    print("===== Solving (HGS / PyVRP) =====")
    t0 = time.time()
    hgs_res = solve_vrpsd_with_hgs(
        inst,
        time_limit_sec=float(TIME_LIMIT_SEC),
        seed=int(HGS_SEED),
        verbose=bool(VERBOSE),
    )
    solve_time = time.time() - t0
    print(f"Done. Solve time: {solve_time:.3f} s\n")

    routes = hgs_res.routes
    det_cost = solution_cost_deterministic(inst, routes)

    # 3) Evaluate with SAA
    print("===== Evaluating (SAA) =====")
    scen_small = ScenarioManager(inst.lam, int(SEED_SMALL)).sample(int(SMALL_S))
    scen_big = ScenarioManager(inst.lam, int(SEED_BIG)).sample(int(BIG_S))

    mean_s, var_s, std_s = saa_mean_var_std(inst, routes, scen_small)
    mean_b, var_b, std_b = saa_mean_var_std(inst, routes, scen_big)
    print("Done.\n")

    # 4) Print results
    print("===== Results =====")
    print(f"Deterministic distance (float Euclidean) : {det_cost:.3f}")
    print(f"Routes (vehicles used)                   : {len(routes)}")
    print(f"PyVRP reported objective (int distances) : {hgs_res.cost:.3f}")
    print(f"PyVRP solve time (sec)                   : {solve_time:.3f}")
    print()
    print(f"[Small SAA] S={SMALL_S}, seed={SEED_SMALL}")
    print(f"  mean = {mean_s:.4f}, var = {var_s:.4f}, std = {std_s:.4f}")
    print(f"[Big   SAA] S={BIG_S}, seed={SEED_BIG}")
    print(f"  mean = {mean_b:.4f}, var = {var_b:.4f}, std = {std_b:.4f}")
    print()

    # 5) Save plots
    if not DISABLE_PLOTS:
        routes_png = os.path.join(out_dir, "hgs_routes.png")
        conv_png = os.path.join(out_dir, "hgs_convergence_alns_style.png")

        plot_routes(inst, routes, routes_png, title=f"HGS routes ({instance_name})")
        plot_convergence_alns_style(
            hgs_res.raw_result,
            conv_png,
            title="Objective (HGS deterministic) over iterations",
            ylabel="Objective (total distance)",
        )

        print("Saved route plot       :", routes_png)
        print("Saved convergence plot :", conv_png)

        if SAVE_NATIVE_CONV:
            native_png = os.path.join(out_dir, "hgs_convergence_pyvrp_native.png")
            plot_convergence_pyvrp_native(hgs_res.raw_result, native_png, title="HGS (PyVRP) objectives")
            print("Saved native conv plot :", native_png)

        print()

    # 6) Save JSON summary
    if SAVE_JSON_SUMMARY:
        summary = {
            "instance": instance_name,
            "nodes_csv": nodes_csv,
            "demand_csv": demand_csv,
            "Q": float(Q),
            "K": int(K),
            "time_limit_sec": float(TIME_LIMIT_SEC),
            "hgs_seed": int(HGS_SEED),
            "small_S": int(SMALL_S),
            "big_S": int(BIG_S),
            "seed_small": int(SEED_SMALL),
            "seed_big": int(SEED_BIG),
            "deterministic_distance": float(det_cost),
            "pyvrp_objective": float(hgs_res.cost),
            "solve_time_sec": float(solve_time),
            "num_routes": int(len(routes)),
            "small_saa": {"mean": float(mean_s), "var": float(var_s), "std": float(std_s)},
            "big_saa": {"mean": float(mean_b), "var": float(var_b), "std": float(std_b)},
            "routes": routes,
        }
        summary_path = os.path.join(out_dir, "hgs_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("Saved summary JSON     :", summary_path)
        print()


def main() -> None:
    # Print parameters at the very top
    print_parameter_console()

    for name in INSTANCES:
        run_one_instance(str(name))


if __name__ == "__main__":
    main()
