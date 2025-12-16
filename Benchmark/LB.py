#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fast LP lower bound for VRPSD (nodes + Poisson lambda) via mean-demand CVRP SCF LP relaxation,
solved by HiGHS through scipy.optimize.linprog (Python 3.13 compatible).

Default params are in CONFIG, but you can override in console:
  python LB.py --nodes ... --demand ... --Q 50 --m 10 --exact-m --timelimit 20
"""

# =======================
# CONFIG (DEFAULTS)
# =======================
CONFIG = {
    "nodes_csv": r"C:\Users\haora\PyCharmMiscProject\datasets\nodes_100_centered.csv",
    "demand_csv": r"C:\Users\haora\PyCharmMiscProject\datasets\demand_100_centered.csv",
    "Q": 50.0,
    "m": 15,
    "exact_m": True,
    "time_limit_sec": 30.0,
    "quiet": False,
}
# =======================

import argparse
import math
import os
import sys
from typing import Dict, Tuple, List

import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def read_data(nodes_csv: str, demand_csv: str) -> Tuple[List[int], Dict[int, Tuple[float, float]], Dict[int, float]]:
    nodes = pd.read_csv(nodes_csv)
    dem = pd.read_csv(demand_csv)

    if not {"node_id", "x", "y"}.issubset(nodes.columns):
        raise ValueError(f"{nodes_csv} must contain columns: node_id,x,y")
    if "node_id" not in dem.columns:
        raise ValueError(f"{demand_csv} must contain column: node_id")

    demand_col = None
    for c in ["lambda", "Lambda", "mean", "demand", "mu"]:
        if c in dem.columns:
            demand_col = c
            break
    if demand_col is None:
        raise ValueError(f"{demand_csv} must contain a demand column, e.g. 'lambda'")

    coords_df = nodes.set_index("node_id")[["x", "y"]]
    coords = {int(i): (float(r["x"]), float(r["y"])) for i, r in coords_df.to_dict("index").items()}

    dser = dem.set_index("node_id")[demand_col]
    d = {int(i): float(v) for i, v in dser.to_dict().items()}

    if 0 not in coords:
        raise ValueError("nodes file must include depot with node_id = 0")
    if 0 not in d:
        d[0] = 0.0

    N = sorted(coords.keys())
    for i in N:
        if i not in d:
            d[i] = 0.0
    return N, coords, d


def solve_lb_highs_scf(
    N: List[int],
    coords: Dict[int, Tuple[float, float]],
    d: Dict[int, float],
    Q: float,
    m: int,
    exact_m: bool,
    time_limit_sec: float,
    quiet: bool,
) -> float:
    depot = 0
    C = [i for i in N if i != depot]

    total_d = sum(d[i] for i in C)
    if total_d <= 0:
        raise ValueError("Total demand must be > 0 (check demand file).")
    if total_d > m * Q + 1e-9:
        print(
            f"[WARN] Sum(mean demands)={total_d:.3f} > m*Q={m*Q:.3f}. "
            "Mean-demand CVRP is infeasible; LP likely infeasible too.",
            file=sys.stderr,
        )

    arcs = [(i, j) for i in N for j in N if i != j]
    A = len(arcs)
    arc_index = {arc: k for k, arc in enumerate(arcs)}

    def x_idx(k: int) -> int:
        return k

    def f_idx(k: int) -> int:
        return A + k

    c = [0.0] * (2 * A)

    def dist(i: int, j: int) -> float:
        xi, yi = coords[i]
        xj, yj = coords[j]
        return math.hypot(xi - xj, yi - yj)

    for k, (i, j) in enumerate(arcs):
        c[x_idx(k)] = dist(i, j)

    bounds = [(0.0, 1.0)] * A + [(0.0, Q)] * A

    # -------- Equalities (COO) --------
    eq_r, eq_c, eq_v = [], [], []
    b_eq = []
    row = 0

    # customer outdegree
    for i in C:
        for j in N:
            if j == i:
                continue
            k = arc_index[(i, j)]
            eq_r.append(row); eq_c.append(x_idx(k)); eq_v.append(1.0)
        b_eq.append(1.0); row += 1

    # customer indegree
    for j in C:
        for i in N:
            if i == j:
                continue
            k = arc_index[(i, j)]
            eq_r.append(row); eq_c.append(x_idx(k)); eq_v.append(1.0)
        b_eq.append(1.0); row += 1

    # depot degree == m if exact
    if exact_m:
        for j in C:
            k = arc_index[(depot, j)]
            eq_r.append(row); eq_c.append(x_idx(k)); eq_v.append(1.0)
        b_eq.append(float(m)); row += 1

        for i in C:
            k = arc_index[(i, depot)]
            eq_r.append(row); eq_c.append(x_idx(k)); eq_v.append(1.0)
        b_eq.append(float(m)); row += 1

    # customer flow: in - out = d_i
    for i in C:
        for j in N:
            if j == i:
                continue
            k_in = arc_index[(j, i)]
            eq_r.append(row); eq_c.append(f_idx(k_in)); eq_v.append(1.0)
            k_out = arc_index[(i, j)]
            eq_r.append(row); eq_c.append(f_idx(k_out)); eq_v.append(-1.0)
        b_eq.append(float(d[i])); row += 1

    # depot flow: out - in = total_d
    for j in C:
        k = arc_index[(depot, j)]
        eq_r.append(row); eq_c.append(f_idx(k)); eq_v.append(1.0)
    for i in C:
        k = arc_index[(i, depot)]
        eq_r.append(row); eq_c.append(f_idx(k)); eq_v.append(-1.0)
    b_eq.append(float(total_d)); row += 1

    A_eq = coo_matrix((eq_v, (eq_r, eq_c)), shape=(row, 2 * A)).tocsr()

    # -------- Inequalities (COO) --------
    ub_r, ub_c, ub_v = [], [], []
    b_ub = []
    row = 0

    # depot degree <= m if not exact
    if not exact_m:
        for j in C:
            k = arc_index[(depot, j)]
            ub_r.append(row); ub_c.append(x_idx(k)); ub_v.append(1.0)
        b_ub.append(float(m)); row += 1

        for i in C:
            k = arc_index[(i, depot)]
            ub_r.append(row); ub_c.append(x_idx(k)); ub_v.append(1.0)
        b_ub.append(float(m)); row += 1

    # linking f_k - Q x_k <= 0
    for k in range(A):
        ub_r.append(row); ub_c.append(f_idx(k)); ub_v.append(1.0)
        ub_r.append(row); ub_c.append(x_idx(k)); ub_v.append(-float(Q))
        b_ub.append(0.0)
        row += 1

    A_ub = coo_matrix((ub_v, (ub_r, ub_c)), shape=(row, 2 * A)).tocsr()

    options = {"time_limit": float(time_limit_sec)}
    if quiet:
        options["log_to_console"] = False

    res = linprog(
        c=c,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options=options,
    )
    if not res.success:
        raise RuntimeError(f"HiGHS failed: status={res.status}, message={res.message}")

    return float(res.fun)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=str, default=None, help="nodes CSV path (overrides CONFIG)")
    ap.add_argument("--demand", type=str, default=None, help="demand CSV path (overrides CONFIG)")
    ap.add_argument("--Q", type=float, default=None, help="vehicle capacity (overrides CONFIG)")
    ap.add_argument("--m", type=int, default=None, help="vehicle count (overrides CONFIG)")
    ap.add_argument("--exact-m", action="store_true", help="force exactly m routes (depot degree == m)")
    ap.add_argument("--no-exact-m", action="store_true", help="force <= m routes (depot degree <= m)")
    ap.add_argument("--timelimit", type=float, default=None, help="HiGHS time limit (seconds)")
    ap.add_argument("--quiet", action="store_true", help="suppress HiGHS log")
    return ap.parse_args()


def main():
    args = parse_args()

    nodes_csv = args.nodes if args.nodes is not None else CONFIG["nodes_csv"]
    demand_csv = args.demand if args.demand is not None else CONFIG["demand_csv"]
    Q = float(args.Q) if args.Q is not None else float(CONFIG["Q"])
    m = int(args.m) if args.m is not None else int(CONFIG["m"])
    time_limit_sec = float(args.timelimit) if args.timelimit is not None else float(CONFIG["time_limit_sec"])
    quiet = bool(args.quiet) if args.quiet else bool(CONFIG["quiet"])

    # exact_m precedence: command line overrides config
    if args.exact_m and args.no_exact_m:
        raise ValueError("Use only one of --exact-m or --no-exact-m")
    if args.exact_m:
        exact_m = True
    elif args.no_exact_m:
        exact_m = False
    else:
        exact_m = bool(CONFIG["exact_m"])

    if not os.path.exists(nodes_csv):
        raise FileNotFoundError(f"nodes_csv not found: {nodes_csv}")
    if not os.path.exists(demand_csv):
        raise FileNotFoundError(f"demand_csv not found: {demand_csv}")

    N, coords, d = read_data(nodes_csv, demand_csv)
    lb = solve_lb_highs_scf(
        N=N, coords=coords, d=d,
        Q=Q, m=m, exact_m=exact_m,
        time_limit_sec=time_limit_sec,
        quiet=quiet,
    )
    print(f"LP lower bound (HiGHS, SCF mean-demand CVRP relaxation) = {lb:.6f}")


if __name__ == "__main__":
    main()
