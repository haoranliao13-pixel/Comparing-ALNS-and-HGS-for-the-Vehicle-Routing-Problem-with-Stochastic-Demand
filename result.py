from __future__ import annotations

import os
import re
import time
import traceback
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
HGS_DIR = os.path.join(ROOT, "HGS_final")
if os.path.isdir(HGS_DIR) and HGS_DIR not in sys.path:
    sys.path.insert(0, HGS_DIR)

CSV_SUBDIR = os.path.join("CVRPLib", "csv")

# Output
OUT_DIR = "outputs"
OUT_CSV = os.path.join(OUT_DIR, "A1_Q_sweep_all_instances.csv")

# Algorithms
RUN_ALNS = True
RUN_HGS = True

# Time limits (seconds)
TIME_LIMIT_ALNS = 20.0
TIME_LIMIT_HGS = 20.0

# ALNS config (keep same for all instances/Q for fair comparison)
ALNS_SEED = 0
ALNS_ITERS = 2000000          
ALNS_NUM_STARTS = 1
ALNS_SMALL_SAMPLES = 80       
ALNS_LARGE_SAMPLES = 300      

# HGS config
HGS_SEED = 0
HGS_VERBOSE = False

# Evaluation (Big SAA you want to report)
BIG_S = 300
BIG_SEED = 54321

# How to pick baseline Q0 from lambdas:
#   Q0_raw = alpha * (sum lambda) / K
Q_ALPHA = 1.25                # safety factor: 1.15~1.35 are common
Q_ROUND_TO = 1               

# Example with percents: [-20%, -10%, 0, +10%, +20%]
Q_DELTAS = [-0.20, -0.10, 0.0, +0.10, +0.20]

# Optional: limit how many instances to run (None = all)
MAX_INSTANCES: Optional[int] = None

# =============================================================================


def find_project_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    for _ in range(12):
        if os.path.isdir(os.path.join(cur, "CVRPLib")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(start_dir)


def list_instances(csv_dir: str) -> List[str]:
    """
    Find all instance names that have both nodes_*.csv and demand_*.csv.
    Returns: ["X-n101-k25", ...]
    """
    inst = []
    for fn in os.listdir(csv_dir):
        if not fn.startswith("nodes_") or not fn.endswith(".csv"):
            continue
        name = fn[len("nodes_") : -len(".csv")]
        dem = f"demand_{name}.csv"
        if os.path.isfile(os.path.join(csv_dir, dem)):
            inst.append(name)
    inst.sort()
    if MAX_INSTANCES is not None:
        inst = inst[: int(MAX_INSTANCES)]
    return inst


def parse_k_from_instance_name(name: str) -> Optional[int]:
    m = re.search(r"-k(\d+)", name)
    return int(m.group(1)) if m else None


def _pick_col(df: pd.DataFrame, *cands: str) -> Optional[str]:
    m = {c.lower().strip(): c for c in df.columns}
    for c in cands:
        if c in m:
            return m[c]
    return None


def read_lambdas_from_demand_csv(demand_csv: str) -> np.ndarray:
    """
    Robustly read lambda vector aligned by id, requiring depot id=0.
    Returns lam array where lam[0]=0.
    """
    dem = pd.read_csv(demand_csv)
    did = _pick_col(dem, "id", "node", "index")
    lmd = _pick_col(dem, "lambda", "lam", "demand", "mu", "mean", "dem")
    if lmd is None:
        raise ValueError(f"Cannot find lambda column in {demand_csv}")

    if did is None:
        dem = dem.assign(id=range(len(dem)))
        did = "id"

    dem = dem.rename(columns={did: "id", lmd: "lambda"})
    dem["id"] = dem["id"].astype(str).str.strip().str.replace("\ufeff", "", regex=False).astype(int)
    dem["lambda"] = pd.to_numeric(dem["lambda"].astype(str).str.strip(), errors="coerce").fillna(0.0)

    dem = dem.sort_values("id").reset_index(drop=True)
    if 0 not in dem["id"].values:
        raise ValueError(f"Depot id=0 not found in {demand_csv}")

    max_id = int(dem["id"].max())
    lam = np.zeros(max_id + 1, dtype=float)
    for _, row in dem.iterrows():
        lam[int(row["id"])] = float(row["lambda"])

    lam[0] = 0.0
    return lam


def round_to(x: float, base: int) -> float:
    if base <= 0:
        return float(x)
    return float(base * round(float(x) / base))


def pick_baseline_Q(lam: np.ndarray, K: int) -> float:
    total = float(np.sum(lam[1:]))
    q0_raw = Q_ALPHA * (total / max(1, int(K)))
    q0 = round_to(q0_raw, Q_ROUND_TO)

    # sanity floor (avoid absurdly small capacity)
    lam_max = float(np.max(lam[1:])) if lam.size > 1 else 0.0
    q0 = max(q0, round_to(lam_max, Q_ROUND_TO))

    # ensure positive
    if q0 <= 0:
        q0 = float(Q_ROUND_TO)
    return float(q0)


def make_Q_levels(q0: float) -> List[float]:
    qs = []
    for d in Q_DELTAS:
        q = round_to(q0 * (1.0 + float(d)), Q_ROUND_TO)
        if q > 0:
            qs.append(float(q))
    # unique + sorted
    qs = sorted(set(qs))
    return qs


def saa_mean_std(instance, routes, S: int, seed: int, normalize_routes_fn, ScenarioManagerCls, recourse_cost_fn) -> Tuple[float, float]:
    scen_mgr = ScenarioManagerCls(instance.lam, seed=seed)
    scenarios = scen_mgr.sample(int(S))

    routes_norm, _ = normalize_routes_fn(routes)
    dist = instance.distance_matrix()

    costs = []
    for dem in scenarios:
        costs.append(float(recourse_cost_fn(instance, routes_norm, dem, dist)))

    arr = np.asarray(costs, dtype=float)
    mean = float(arr.mean()) if arr.size else float("nan")
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def safe_len_routes(routes) -> int:
    return int(sum(1 for r in routes if r))


def main() -> None:
    root = find_project_root(os.path.dirname(os.path.abspath(__file__)))
    csv_dir = os.path.join(root, CSV_SUBDIR)
    os.makedirs(os.path.join(root, OUT_DIR), exist_ok=True)

    instances = list_instances(csv_dir)
    if not instances:
        raise RuntimeError(f"No instances found under: {csv_dir}")

    print("===== A1 Batch Experiment: Fix K (=k in instance name), sweep Q =====")
    print("Project root :", root)
    print("CSV dir      :", csv_dir)
    print("Instances    :", len(instances))
    print(f"Time limits  : ALNS={TIME_LIMIT_ALNS}s, HGS={TIME_LIMIT_HGS}s")
    print(f"Big SAA      : S={BIG_S}, seed={BIG_SEED}")
    print(f"Q baseline   : Q0 = {Q_ALPHA} * sum(lambda)/K, rounded to {Q_ROUND_TO}")
    print(f"Q deltas     : {Q_DELTAS}")
    print("=====================================================================\n")

    # ---- import ALNS modules (prefer package import alns2.*) ----
    if RUN_ALNS:
        try:
            from alns2.problem import build_instance_from_csv as build_alns_instance
            from alns2.problem import normalize_routes as alns_normalize_routes
            from alns2.solver import ALNSSolveConfig, solve_vrpsd_with_alns
            from alns2.evaluator import ScenarioManager as ALNSScenarioManager
            from alns2.evaluator import recourse_cost_one_scenario as alns_recourse_cost
        except Exception as e:
            raise RuntimeError(
                "Failed to import ALNS modules as package 'alns2.*'.\n"
                "Make sure your folder structure is:\n"
                "  <root>/alns2/__init__.py\n"
                "  <root>/alns2/problem.py, solver.py, evaluator.py, ...\n"
                f"Original error: {e}"
            )

        alns_cfg = ALNSSolveConfig(
            seed=ALNS_SEED,
            time_limit_sec=float(TIME_LIMIT_ALNS),
            iters=int(ALNS_ITERS),
            small_samples=int(ALNS_SMALL_SAMPLES),
            large_samples=int(ALNS_LARGE_SAMPLES),
            num_starts=int(ALNS_NUM_STARTS),
            use_adaptive_selection=True,
        )

    # ---- import HGS modules (prefer package import HGS_final.*) ----
    if RUN_HGS:
        try:
            from HGS_final.problem import build_instance_from_csv as build_hgs_instance
            from HGS_final.solver import solve_vrpsd_with_hgs
        except Exception as e:
            raise RuntimeError(
                "Failed to import HGS modules as package 'HGS_final.*'.\n"
                "Make sure your folder structure is:\n"
                "  <root>/HGS_final/__init__.py\n"
                "  <root>/HGS_final/problem.py, solver.py, evaluator.py, visualize.py, run.py\n"
                f"Original error: {e}"
            )

        # For evaluation we reuse ALNS evaluator for consistency
        # (so mean/std are computed in exactly the same way for both algorithms).
        if not RUN_ALNS:
            # if user disables ALNS but still wants evaluation functions,
            # we still import alns2.evaluator + normalize_routes for consistent evaluation.
            from alns2.problem import normalize_routes as alns_normalize_routes
            from alns2.evaluator import ScenarioManager as ALNSScenarioManager
            from alns2.evaluator import recourse_cost_one_scenario as alns_recourse_cost

    rows: List[Dict] = []

    for idx, inst_name in enumerate(instances, start=1):
        k = parse_k_from_instance_name(inst_name)
        if k is None:
            print(f"[SKIP] Cannot parse k from instance name: {inst_name}")
            continue
        K = int(k)

        nodes_csv = os.path.join(csv_dir, f"nodes_{inst_name}.csv")
        demand_csv = os.path.join(csv_dir, f"demand_{inst_name}.csv")

        try:
            lam = read_lambdas_from_demand_csv(demand_csv)
            q0 = pick_baseline_Q(lam, K=K)
            q_levels = make_Q_levels(q0)
        except Exception as e:
            print(f"[ERROR] Failed to prepare Q levels for {inst_name}: {e}")
            continue

        print(f"----- ({idx}/{len(instances)}) Instance: {inst_name} | K={K} | Q0={q0} | Qs={q_levels} -----")

        for Q in q_levels:
            # ---------------- ALNS ----------------
            if RUN_ALNS:
                try:
                    inst = build_alns_instance(nodes_csv, demand_csv, Q=float(Q), K=int(K))

                    t0 = time.perf_counter()
                    alns_res = solve_vrpsd_with_alns(inst, alns_cfg)
                    t_alns = time.perf_counter() - t0

                    mean_big, std_big = saa_mean_std(
                        inst,
                        alns_res.routes,
                        S=BIG_S,
                        seed=BIG_SEED,
                        normalize_routes_fn=alns_normalize_routes,
                        ScenarioManagerCls=ALNSScenarioManager,
                        recourse_cost_fn=alns_recourse_cost,
                    )

                    rnum = safe_len_routes(alns_res.routes)

                    row = {
                        "instance": inst_name,
                        "algorithm": "ALNS",
                        "Q": float(Q),
                        "K": int(K),
                        "time_sec": float(t_alns),
                        "big_saa_mean": float(mean_big),
                        "big_saa_std": float(std_big),
                        "routes": int(rnum),
                        "status": "ok",
                    }
                    rows.append(row)

                    print(f"  ALNS | Q={Q:8.2f} | time={t_alns:7.2f}s | routes={rnum:3d} | mean={mean_big:10.4f} | std={std_big:9.4f}")

                except Exception as e:
                    rows.append({
                        "instance": inst_name,
                        "algorithm": "ALNS",
                        "Q": float(Q),
                        "K": int(K),
                        "time_sec": float("nan"),
                        "big_saa_mean": float("nan"),
                        "big_saa_std": float("nan"),
                        "routes": -1,
                        "status": f"error: {e}",
                    })
                    print(f"  [ALNS ERROR] Q={Q} -> {e}")
                    
            # ---------------- HGS ----------------
            if RUN_HGS:
                try:
                    inst_h = build_hgs_instance(nodes_csv, demand_csv, Q=float(Q), K=int(K))

                    t0 = time.perf_counter()
                    hgs_res = solve_vrpsd_with_hgs(
                        inst_h,
                        time_limit_sec=float(TIME_LIMIT_HGS),
                        seed=int(HGS_SEED),
                        verbose=bool(HGS_VERBOSE),
                    )
                    t_hgs = time.perf_counter() - t0

                    mean_big, std_big = saa_mean_std(
                        inst_h,
                        hgs_res.routes,
                        S=BIG_S,
                        seed=BIG_SEED,
                        normalize_routes_fn=alns_normalize_routes,
                        ScenarioManagerCls=ALNSScenarioManager,
                        recourse_cost_fn=alns_recourse_cost,
                    )

                    rnum = safe_len_routes(hgs_res.routes)

                    row = {
                        "instance": inst_name,
                        "algorithm": "HGS",
                        "Q": float(Q),
                        "K": int(K),
                        "time_sec": float(t_hgs),
                        "big_saa_mean": float(mean_big),
                        "big_saa_std": float(std_big),
                        "routes": int(rnum),
                        "status": "ok",
                    }
                    rows.append(row)

                    print(f"  HGS  | Q={Q:8.2f} | time={t_hgs:7.2f}s | routes={rnum:3d} | mean={mean_big:10.4f} | std={std_big:9.4f}")

                except Exception as e:
                    rows.append({
                        "instance": inst_name,
                        "algorithm": "HGS",
                        "Q": float(Q),
                        "K": int(K),
                        "time_sec": float("nan"),
                        "big_saa_mean": float("nan"),
                        "big_saa_std": float("nan"),
                        "routes": -1,
                        "status": f"error: {e}",
                    })
                    print(f"  [HGS ERROR] Q={Q} -> {e}")
                    # traceback.print_exc()

        print()

    df = pd.DataFrame(rows)

    # Nice ordering
    cols = ["instance", "algorithm", "Q", "K", "time_sec", "big_saa_mean", "big_saa_std", "routes", "status"]
    df = df[cols].sort_values(["instance", "algorithm", "Q"]).reset_index(drop=True)

    out_path = os.path.join(root, OUT_CSV)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    print("===== DONE =====")
    print("Rows:", len(df))
    print("Saved CSV:", out_path)


if __name__ == "__main__":
    main()
