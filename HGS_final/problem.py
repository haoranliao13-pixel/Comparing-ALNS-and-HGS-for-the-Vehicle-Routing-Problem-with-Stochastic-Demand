from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd


Route = List[int]
Solution = List[Route]


@dataclass
class VRPSDInstance:
    """
    Minimal VRPSD instance for HGS side.

    coords : (n,2) array of node coordinates (0 = depot)
    lam    : (n,)  array of expected demands (lam[0] must be 0)
    Q      : vehicle capacity
    K      : optional max number of vehicles
    """
    coords: np.ndarray
    lam: np.ndarray
    Q: float
    K: Optional[int] = None
    _D: Optional[np.ndarray] = None  # lazy cache

    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords, dtype=float)
        self.lam = np.asarray(self.lam, dtype=float)

        if self.coords.ndim != 2 or self.coords.shape[1] != 2:
            raise ValueError("coords must be shape (n,2).")
        if self.coords.shape[0] != self.lam.shape[0]:
            raise ValueError("coords and lam mismatch.")

        # depot lambda forced to 0
        if self.lam.shape[0] > 0:
            self.lam[0] = 0.0

    def distance_matrix(self) -> np.ndarray:
        """Lazy Euclidean distance matrix."""
        if self._D is None:
            diff = self.coords[:, None, :] - self.coords[None, :, :]
            self._D = np.hypot(diff[..., 0], diff[..., 1])
        return self._D


def _pick_col(df: pd.DataFrame, *cands: str) -> Optional[str]:
    """Pick first existing column among candidates (case-insensitive)."""
    m = {c.lower().strip(): c for c in df.columns}
    for c in cands:
        if c in m:
            return m[c]
    return None


def build_instance_from_csv(
    nodes_csv: str,
    demand_csv: str,
    Q: float,
    K: Optional[int] = None,
) -> VRPSDInstance:
    """
    CVRPLib-style reader.

    nodes_csv:
      - must contain x/y columns (or xcoord/ycoord)
      - id is optional; if missing, it is assumed 0..n-1
    demand_csv:
      - must contain a demand column among: lambda/lam/demand/mu/mean/dem
      - id is optional; if missing, it is assumed 0..n-1

    Requirements:
      - depot id must be 0
      - lam[0] is forced to 0
      - missing lambda for non-depot nodes will be warned and filled with 0
    """
    nodes = pd.read_csv(nodes_csv)
    dem = pd.read_csv(demand_csv)

    nid = _pick_col(nodes, "id", "node", "index")
    xid = _pick_col(nodes, "x", "xcoord")
    yid = _pick_col(nodes, "y", "ycoord")

    did = _pick_col(dem, "id", "node", "index")
    lmd = _pick_col(dem, "lambda", "lam", "demand", "mu", "mean", "dem")

    if xid is None or yid is None:
        raise ValueError("nodes CSV must contain x,y columns (or xcoord/ycoord).")
    if lmd is None:
        raise ValueError(
            "demand CSV must contain a demand column: one of "
            "[lambda, lam, demand, mu, mean, dem]."
        )

    # If no id, assume 0..n-1
    if nid is None:
        nodes = nodes.assign(id=range(len(nodes)))
        nid = "id"
    if did is None:
        dem = dem.assign(id=range(len(dem)))
        did = "id"

    nodes = nodes.rename(columns={nid: "id", xid: "x", yid: "y"})
    dem = dem.rename(columns={did: "id", lmd: "lambda"})

    merged = nodes.merge(dem, on="id", how="left")
    merged["id"] = merged["id"].astype(int)
    merged = merged.sort_values("id").reset_index(drop=True)

    if 0 not in merged["id"].values:
        raise ValueError("Require a depot with id=0 in nodes/demand CSVs.")

    # depot lambda = 0
    merged.loc[merged["id"] == 0, "lambda"] = merged.loc[
        merged["id"] == 0, "lambda"
    ].fillna(0.0)

    # warn missing lambda for customers, then fill 0
    bad = merged["lambda"].isna() & (merged["id"] != 0)
    if bad.any():
        miss_ids = merged.loc[bad, "id"].astype(int).tolist()
        print(f"[WARN] lambda missing for ids {miss_ids} -> filled 0.0")
        merged.loc[bad, "lambda"] = 0.0

    merged = merged.sort_values("id").reset_index(drop=True)
    coords = merged[["x", "y"]].to_numpy(float)
    lam = merged["lambda"].to_numpy(float)
    lam[0] = 0.0

    return VRPSDInstance(coords=coords, lam=lam, Q=float(Q), K=K)


def solution_cost_deterministic(
    instance: VRPSDInstance,
    routes: Solution,
    dist_matrix: Optional[np.ndarray] = None,
) -> float:
    """Deterministic total travel distance (ignore stochasticity)."""
    if dist_matrix is None:
        dist_matrix = instance.distance_matrix()
    tot = 0.0
    for r in routes:
        prev = 0
        for c in r:
            tot += float(dist_matrix[prev, c])
            prev = c
        tot += float(dist_matrix[prev, 0])
    return float(tot)
