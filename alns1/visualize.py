from typing import List
import numpy as np
import matplotlib.pyplot as plt

try:
    from .problem import VRPSDInstance, Solution
except ImportError:
    from problem import VRPSDInstance, Solution

def plot_routes(instance: VRPSDInstance, routes: Solution, fname: str = None, title: str = "Final routes"):
    coords = instance.coords
    plt.figure()
    plt.scatter([coords[0,0]], [coords[0,1]], marker='s', s=80, label='Depot')
    plt.scatter(coords[1:,0], coords[1:,1], marker='o', s=30, label='Customers')
    for r in routes:
        if not r: continue
        seq = [0]+r+[0]
        xs, ys = coords[seq,0], coords[seq,1]
        plt.plot(xs, ys, linewidth=1.5)
    plt.legend(); plt.title(title)
    if fname: plt.savefig(fname, bbox_inches='tight')
    return plt.gcf()

def plot_convergence(cost_history: List[float], fname: str = None, title: str = "Expected cost (small-sample) over iterations"):
    import numpy as np
    plt.figure()
    plt.plot(np.arange(1, len(cost_history)+1), cost_history, linewidth=1.5)
    plt.xlabel("Iteration"); plt.ylabel("Expected cost (SAA small sample)")
    plt.title(title)
    if fname: plt.savefig(fname, bbox_inches='tight')
    return plt.gcf()
