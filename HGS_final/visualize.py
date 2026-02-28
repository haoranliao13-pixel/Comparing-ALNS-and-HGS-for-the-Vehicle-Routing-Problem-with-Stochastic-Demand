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




def plot_convergence_pyvrp_native(
    pyvrp_result,
    out_path: str,
    title: str = "HGS (PyVRP) objectives",
) -> None:
   
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



