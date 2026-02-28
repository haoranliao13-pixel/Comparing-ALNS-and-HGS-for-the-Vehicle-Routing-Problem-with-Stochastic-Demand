
from typing import List
import random
import numpy as np

try:
    from .problem import Solution, Route
except ImportError:
    from problem import Solution, Route

def random_removal(sol: Solution, num_remove: int, rng: random.Random) -> Solution:
    customers = [c for r in sol for c in r]
    if len(customers) <= num_remove:
        return [[]]
    removed = set(rng.sample(customers, num_remove))
    new_routes: Solution = []
    for r in sol:
        nr = [c for c in r if c not in removed]
        if nr: new_routes.append(nr)
    return new_routes

def worst_removal(sol: Solution, num_remove: int, distances: np.ndarray, rng: random.Random) -> Solution:
    contribs = []
    for r in sol:
        prev = 0
        for i, c in enumerate(r):
            nxt = r[i+1] if i+1 < len(r) else 0
            gain = distances[prev, c] + distances[c, nxt] - distances[prev, nxt]
            contribs.append((gain, c)); prev = c
    contribs.sort(reverse=True)
    remove = set(c for _, c in contribs[:min(num_remove, len(contribs))])
    new_routes: Solution = []
    for r in sol:
        nr = [c for c in r if c not in remove]
        if nr: new_routes.append(nr)
    return new_routes

def shaw_removal(sol: Solution, num_remove: int, distances: np.ndarray, rng: random.Random) -> Solution:
    customers = [c for r in sol for c in r]
    if not customers: return sol
    seed = rng.choice(customers); removed = {seed}
    while len(removed) < min(num_remove, len(customers)):
        best, best_score = None, float("inf")
        for c in customers:
            if c in removed: continue
            sc = min(distances[c, r] for r in removed)
            if sc < best_score: best_score, best = sc, c
        removed.add(best)
    new_routes: Solution = []
    for r in sol:
        nr = [c for c in r if c not in removed]
        if nr: new_routes.append(nr)
    return new_routes

def greedy_insert(sol: Solution, removed: List[int], distances: np.ndarray, rng: random.Random) -> Solution:
    routes = [list(r) for r in sol]
    for c in removed:
        if not routes: routes.append([c]); continue
        best_route, best_pos, best_delta = 0, 0, float("inf")
        for r_idx, r in enumerate(routes):
            prev = 0
            for i in range(len(r)+1):
                nxt = r[i] if i < len(r) else 0
                delta = distances[prev, c] + distances[c, nxt] - distances[prev, nxt]
                if delta < best_delta: best_delta, best_route, best_pos = delta, r_idx, i
                prev = nxt if i < len(r) else 0
        routes[best_route].insert(best_pos, c)
    return routes

def regret_k_insert(sol: Solution, removed: List[int], distances: np.ndarray, k: int, rng: random.Random) -> Solution:
    routes = [list(r) for r in sol]
    for c in removed:
        costs = []; pos = []
        if not routes: routes = [[]]
        for r_idx, r in enumerate(routes):
            best_delta, best_pos = float("inf"), 0
            prev = 0
            for i in range(len(r)+1):
                nxt = r[i] if i < len(r) else 0
                delta = distances[prev, c] + distances[c, nxt] - distances[prev, nxt]
                if delta < best_delta: best_delta, best_pos = delta, i
                prev = nxt if i < len(r) else 0
            costs.append(best_delta); pos.append((r_idx, best_pos))
        idx = int(np.argmin(costs))
        r_idx, p = pos[idx]
        routes[r_idx].insert(p, c)
    return routes

def removed_customers(before: Solution, after: Solution) -> List[int]:
    B = [c for r in before for c in r]
    A = [c for r in after  for c in r]
    return [c for c in B if c not in A]

# --------------- Cross-route local search operators ---------------
def relocate_10(routes: Solution, distances: np.ndarray, max_trials: int = 200) -> Solution:
    """Move a single customer between routes if distance decreases."""
    import random
    rng = random.Random(0)
    routes = [list(r) for r in routes]
    if sum(len(r) for r in routes) <= 1:
        return routes
    for _ in range(max_trials):
        if not routes: break
        i = rng.randrange(len(routes))
        j = rng.randrange(len(routes))
        if i == j or not routes[i]: 
            continue
        ri, rj = routes[i], routes[j]
        p = rng.randrange(len(ri))
        c = ri[p]
        # remove cost saving from ri
        prev_i = ri[p-1] if p > 0 else 0
        nxt_i  = ri[p+1] if p+1 < len(ri) else 0
        save = distances[prev_i, c] + distances[c, nxt_i] - distances[prev_i, nxt_i]
        # best insertion in rj
        best_pos, best_delta = 0, float("inf")
        prev = 0
        for q in range(len(rj)+1):
            nxt = rj[q] if q < len(rj) else 0
            delta = distances[prev, c] + distances[c, nxt] - distances[prev, nxt]
            if delta < best_delta:
                best_delta, best_pos = float(delta), q
            prev = nxt if q < len(rj) else 0
        if best_delta - save < -1e-9:
            del ri[p]
            rj.insert(best_pos, c)
            # drop empty
            routes = [r for r in routes if len(r) > 0]
    return routes

def two_opt_star(routes: Solution, distances: np.ndarray, max_trials: int = 300) -> Solution:
    """2-opt* between two routes to reduce crossings (distance-based)."""
    import random
    rng = random.Random(1)
    routes = [list(r) for r in routes]
    if len(routes) < 2:
        return routes
    for _ in range(max_trials):
        a = rng.randrange(len(routes))
        b = rng.randrange(len(routes))
        if a == b or len(routes[a]) < 2 or len(routes[b]) < 2:
            continue
        ra, rb = routes[a], routes[b]
        i = rng.randrange(0, len(ra)-1)
        j = rng.randrange(0, len(rb)-1)
        a1, a2 = ra[i], ra[i+1]
        b1, b2 = rb[j], rb[j+1]
        old = distances[a1, a2] + distances[b1, b2]
        new = distances[a1, b1] + distances[a2, b2]
        if new + 1e-9 < old:
            # reverse suffixes to realize the new edges
            ra[i+1:] = reversed(ra[i+1:])
            rb[j+1:] = reversed(rb[j+1:])
    return [r for r in routes if len(r) > 0]

def or_opt2(routes: Solution, distances: np.ndarray, max_trials: int = 200) -> Solution:
    """Move a consecutive pair between routes if it improves distance."""
    import random
    rng = random.Random(2)
    routes = [list(r) for r in routes]
    for _ in range(max_trials):
        if not routes: break
        i = rng.randrange(len(routes))
        j = rng.randrange(len(routes))
        if i == j or len(routes[i]) < 2:
            continue
        ri, rj = routes[i], routes[j]
        if len(ri) < 2:
            continue
        p = rng.randrange(0, len(ri)-1)
        seg = ri[p:p+2]
        # remove from ri saving
        prev_i = ri[p-1] if p > 0 else 0
        nxt_i  = ri[p+2] if p+2 < len(ri) else 0
        save = distances[prev_i, seg[0]] + distances[seg[1], nxt_i] - distances[prev_i, nxt_i]
        # best insertion into rj
        best_pos, best_delta = 0, float("inf")
        prev = 0
        for q in range(len(rj)+1):
            nxt = rj[q] if q < len(rj) else 0
            delta = distances[prev, seg[0]] + distances[seg[1], nxt] - distances[prev, nxt]
            if delta < best_delta:
                best_delta, best_pos = float(delta), q
            prev = nxt if q < len(rj) else 0
        if best_delta - save < -1e-9:
            del ri[p:p+2]
            for k, x in enumerate(seg):
                rj.insert(best_pos + k, x)
            routes = [r for r in routes if len(r) > 0]
    return routes
