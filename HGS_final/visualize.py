from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt

from problem import VRPSDInstance, Solution


def plot_routes(
    instance: VRPSDInstance,
    routes: Solution,
    out_path: str,
    title: str = "HGS routes",
) -> None:
    """
    Save a route plot to out_path.

    Style intentionally close to ALNS visualize: scatter nodes, polyline per route.
    """
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    coords = instance.coords

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(coords[1:, 0], coords[1:, 1], s=10, label="Customers")
    ax.scatter(coords[0, 0], coords[0, 1], s=40, label="Depot")

    for r in routes:
        seq = [0] + list(r) + [0]
        ax.plot(coords[seq, 0], coords[seq, 1], linewidth=1)

    ax.set_title(title)
    ax.set_aspect("equal", "box")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_convergence_alns_style(
    pyvrp_result,
    out_path: str,
    title: str = "Objective (HGS deterministic) over iterations",
    ylabel: str = "Objective (total distance)",
) -> None:
    """
    Save an ALNS-like convergence curve (single line over iterations).

    We extract the 'Best' curve from pyvrp.plotting.plot_objectives and re-plot it
    as a single-line chart.

    NOTE: metric is PyVRP deterministic objective, NOT SAA expected cost.
    """
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        from pyvrp.plotting import plot_objectives

        # 1) Draw to a temporary axis so we can access the line data
        fig_tmp, ax_tmp = plt.subplots()
        plot_objectives(pyvrp_result, ax=ax_tmp)

        best_line = None
        for ln in ax_tmp.lines:
            label = (ln.get_label() or "").strip().lower()
            if label.startswith("best"):
                best_line = ln
                break

        if best_line is None:
            plt.close(fig_tmp)
            raise RuntimeError("Could not find 'Best' line in PyVRP objective plot.")

        x = best_line.get_xdata()
        y = best_line.get_ydata()
        plt.close(fig_tmp)

        # 2) Re-plot in ALNS style: single line
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, y, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    except Exception as e:
        print("[WARN] plot_convergence_alns_style failed:", e)


def plot_convergence_pyvrp_native(
    pyvrp_result,
    out_path: str,
    title: str = "HGS (PyVRP) objectives",
) -> None:
    """
    Optional: Save the native PyVRP convergence plot (current/candidate/best).
    Kept for debugging, but not used if you want ALNS-like format.
    """
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        from pyvrp.plotting import plot_objectives

        fig, ax = plt.subplots(figsize=(6, 4))
        plot_objectives(pyvrp_result, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective (total distance)")
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    except Exception as e:
        print("[WARN] plot_convergence_pyvrp_native failed:", e)
