
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class VRPSDInstance:
    coords: np.ndarray          # shape (n, 2), node 0 is depot
    lam: np.ndarray             # shape (n,), expected demand; lam[0] = 0
    Q: float
    K: Optional[int] = None

    def __post_init__(self):
        assert self.coords.shape[0] == self.lam.shape[0], "coords and lam mismatch"
        assert self.coords.shape[1] == 2, "coords must be (n,2)"
        assert self.lam[0] == 0.0, "depot demand must be 0"

    @property
    def n_customers(self) -> int:
        return self.coords.shape[0] - 1

    def dist(self, i: int, j: int) -> float:
        return float(np.linalg.norm(self.coords[i] - self.coords[j]))

    def distance_matrix(self) -> np.ndarray:
        a = self.coords[:, None, :]
        b = self.coords[None, :, :]
        return np.sqrt(((a - b) ** 2).sum(axis=2))


def _pick(df, *cands):
    m = {c.lower().strip(): c for c in df.columns}
    for c in cands:
        if c in m:
            return m[c]
    return None


def build_instance_from_csv(nodes_csv: str, demand_csv: str, Q: float, K: Optional[int] = None) -> VRPSDInstance:
    """
    Robust reader:
    - nodes: expects id,x,y (accepts node/index as id alias; generates sequential id if absent)
    - demand: expects id,lambda (accepts lam/demand/mu/mean alias; generates sequential id if absent)
    - trims spaces/BOM, coerces lambda to numeric; missing/NaN non-depot lambda -> warn and fill 0.0
    - requires depot id=0 as first row after sorting
    """
    import pandas as pd

    nodes = pd.read_csv(nodes_csv)
    dem   = pd.read_csv(demand_csv)

    nid = _pick(nodes, "id","node","index")
    xid = _pick(nodes, "x")
    yid = _pick(nodes, "y")
    did = _pick(dem,   "id","node","index")
    lmd = _pick(dem,   "lambda","lam","demand","mu","mean","dem")

    if xid is None or yid is None:
        raise ValueError("nodes_300_corner.csv must contain x,y columns")
    if lmd is None:
        raise ValueError("demand_300_corner.csv must contain demand column: one of [lambda, lam, demand, mu, mean, dem]")

    if nid is None:
        nodes = nodes.assign(id=range(len(nodes))); nid = "id"
    if did is None:
        dem   = dem.assign(id=range(len(dem)));     did = "id"

    nodes = nodes.rename(columns={nid:"id", xid:"x", yid:"y"})
    dem   = dem.rename(columns={did:"id", lmd:"lambda"})

    # Clean ids
    nodes["id"] = nodes["id"].astype(str).str.strip().str.replace("\ufeff","", regex=False).astype(int)
    dem["id"]   = dem["id"].astype(str).str.strip().str.replace("\ufeff","", regex=False).astype(int)

    # Drop duplicate demand ids (keep first)
    dem = dem[~dem["id"].duplicated(keep="first")]

    # Coerce lambda
    dem["lambda"] = pd.to_numeric(dem["lambda"].astype(str).str.strip(), errors="coerce")

    merged = nodes.merge(dem, on="id", how="left", sort=True, validate="one_to_one")

    # Depot fill
    merged.loc[merged["id"] == 0, "lambda"] = merged.loc[merged["id"] == 0, "lambda"].fillna(0.0)

    # Others: warn and fill 0
    bad = merged["lambda"].isna() & (merged["id"] != 0)
    if bad.any():
        miss_ids = merged.loc[bad, "id"].astype(int).tolist()
        print(f"[WARN] lambda missing/unparseable for ids: {miss_ids} -> filled 0.0")
        merged.loc[bad, "lambda"] = 0.0

    merged = merged.sort_values("id").reset_index(drop=True)

    # enforce depot id 0 at top
    if merged.loc[0, "id"] != 0:
        raise ValueError("Require depot id=0 and sorted ascending by id.")

    coords = merged[["x","y"]].to_numpy(float)
    lam    = merged["lambda"].to_numpy(float)
    lam[0] = 0.0

    return VRPSDInstance(coords=coords, lam=lam, Q=Q, K=K)


Route = List[int]
Solution = List[Route]


def solution_cost_deterministic(instance: VRPSDInstance, routes: Solution, dist_matrix: Optional[np.ndarray] = None) -> float:
    if dist_matrix is None:
        dist_matrix = instance.distance_matrix()
    tot = 0.0
    for r in routes:
        prev = 0
        for c in r:
            tot += dist_matrix[prev, c]
            prev = c
        tot += dist_matrix[prev, 0]
    return tot


def normalize_routes(routes: Solution):
    def canon(route):
        if not route:
            return tuple()
        m = min(route); k = route.index(m)
        rot = route[k:] + route[:k]
        return tuple(rot)
    key = tuple(sorted(canon(r) for r in routes if r))
    return [list(t) for t in key], key


def greedy_split_by_capacity(instance: VRPSDInstance, permutation: List[int]) -> Solution:
    Q = instance.Q; lam = instance.lam
    routes: Solution = []
    cur: Route = []; load = 0.0
    for c in permutation:
        if c == 0: continue
        if cur and load + lam[c] > Q:
            routes.append(cur); cur = [c]; load = lam[c]
        else:
            cur.append(c); load += lam[c]
    if cur: routes.append(cur)
    return routes


def k_min_lower_bound(instance: VRPSDInstance) -> int:
    return int(np.ceil(float(instance.lam.sum()) / instance.Q))


def _merge_two_routes_cost(dist, A, B):
    """Return (delta_cost, A_out, B_out, mode) for best way to connect A and B (consider reversing)."""
    # Consider four variants: A(+/-), B(+/-). We'll try connect end->start.
    best = (float("inf"), None, None, None)
    variantsA = [A, list(reversed(A))]
    variantsB = [B, list(reversed(B))]
    for ai, AA in enumerate(variantsA):
        for bi, BB in enumerate(variantsB):
            # original separate cost contribution to depot:
            sep = dist[AA[-1], 0] + dist[0, BB[0]]
            # merged bridge:
            bridge = dist[AA[-1], BB[0]]
            delta = bridge - sep
            mode = (ai, bi)
            if delta < best[0]:
                best = (delta, AA, BB, mode)
    return best  # delta, Aseq, Bseq, mode


def limit_routes_to_K(instance: VRPSDInstance, routes: Solution, K: Optional[int]) -> Solution:
    """If number of routes exceeds K, iteratively merge two routes with minimal cost increase (or maximal saving)."""
    if K is None or K <= 0:
        return routes
    routes = [list(r) for r in routes if len(r) > 0]
    if len(routes) <= K:
        return routes
    D = instance.distance_matrix()
    # Greedy merging until count == K
    while len(routes) > K:
        best_pair = None
        best_delta = float("inf")
        best_variants = None
        m = len(routes)
        for i in range(m):
            for j in range(m):
                if i == j: continue
                di, Ai, Bi, mode = _merge_two_routes_cost(D, routes[i], routes[j])
                if di < best_delta:
                    best_delta = di; best_pair = (i, j); best_variants = (Ai, Bi, mode)
        i, j = best_pair
        Ai, Bi, _ = best_variants
        merged = Ai + Bi
        # rebuild new list
        new_routes = []
        for t, r in enumerate(routes):
            if t == i or t == j: continue
            new_routes.append(r)
        new_routes.append(merged)
        routes = new_routes
    return routes