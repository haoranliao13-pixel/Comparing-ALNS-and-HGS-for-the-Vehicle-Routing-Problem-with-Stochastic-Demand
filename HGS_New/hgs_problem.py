from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class VRPSDInstance:
    """
    Minimal VRPSD instance for HGS side.

    coords : (n,2) array of node coordinates (0 = depot)
    lam    : (n,)  array of expected demands (lam[0] should be 0)
    Q      : vehicle capacity
    K      : optional max number of vehicles
    """
    coords: np.ndarray
    lam: np.ndarray
    Q: float
    K: Optional[int] = None
    _D: Optional[np.ndarray] = None  # lazy cache

    def __post_init__(self) -> None:
        assert self.coords.shape[0] == self.lam.shape[0], "coords and lam mismatch"
        assert self.coords.shape[1] == 2, "coords must be (n,2)"

    def distance_matrix(self) -> np.ndarray:
        """Lazy Euclidean distance matrix."""
        if self._D is None:
            diff = self.coords[:, None, :] - self.coords[None, :, :]
            self._D = np.hypot(diff[..., 0], diff[..., 1])
        return self._D


Route = List[int]
Solution = List[Route]


def _pick(df: pd.DataFrame, *cands: str) -> Optional[str]:
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
    Simple CVRPLib-style reader.

    - nodes_csv: columns for id / x / y (id 可以缺省，自动 0..n-1)
    - demand_csv: 一列需求，列名可以是 lambda / lam / demand / mu / mean / dem
    - 要求 depot 的 id = 0，lam[0] 强制设为 0
    """
    nodes = pd.read_csv(nodes_csv)
    dem = pd.read_csv(demand_csv)

    def pick(df: pd.DataFrame, *cands: str) -> Optional[str]:
        m = {c.lower().strip(): c for c in df.columns}
        for c in cands:
            if c in m:
                return m[c]
        return None

    nid = pick(nodes, "id", "node", "index")
    xid = pick(nodes, "x", "xcoord")
    yid = pick(nodes, "y", "ycoord")
    did = pick(dem, "id", "node", "index")
    lmd = pick(dem, "lambda", "lam", "demand", "mu", "mean", "dem")

    if xid is None or yid is None:
        raise ValueError("nodes CSV must contain x,y columns (or XCOORD,YCOORD).")
    if lmd is None:
        raise ValueError(
            "demand CSV must contain a demand column: one of "
            "[lambda, lam, demand, mu, mean, dem]."
        )

    # 如果没 id，就补一个 0..n-1
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

    # depot λ = 0
    merged.loc[merged["id"] == 0, "lambda"] = merged.loc[
        merged["id"] == 0, "lambda"
    ].fillna(0.0)

    # 其他缺失的 λ 先警告，再补 0
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
            tot += dist_matrix[prev, c]
            prev = c
        tot += dist_matrix[prev, 0]
    return float(tot)
