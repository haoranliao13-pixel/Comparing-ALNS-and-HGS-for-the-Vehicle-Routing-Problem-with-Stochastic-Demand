
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