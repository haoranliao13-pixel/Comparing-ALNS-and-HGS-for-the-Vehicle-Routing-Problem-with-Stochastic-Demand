
from dataclasses import dataclass, field
from typing import List
import random
import numpy as np
import time

from alns import ALNS, State, Result
from alns.accept import SimulatedAnnealing
from alns.stop import MaxIterations, NoImprovement
from alns.select import RandomSelect, RouletteWheel

try:
    from alns.stop import Or  # ALNS 7.x
except Exception:
    Or = None  # fallback

# Project imports (package/flat both supported)
try:
    from .problem import VRPSDInstance, Solution, greedy_split_by_capacity, limit_routes_to_K
    from .evaluator import ScenarioManager, SAACache, evaluate_routes
except ImportError:
    from problem import VRPSDInstance, Solution, greedy_split_by_capacity, limit_routes_to_K
    from evaluator import ScenarioManager, SAACache, evaluate_routes


# -----------------------------
# State (ALNS 7.0: objective() must exist)
# -----------------------------
class VRPSDState(State):
    def __init__(self, routes: 'Solution', objective_fn):
        self.routes = [list(r) for r in routes]
        self._objective = objective_fn  # callable(state) -> float
        # optional attribute set by destroy: list of removed customers
        self._removed = None

    def copy(self) -> "VRPSDState":
        cp = VRPSDState([list(r) for r in self.routes], self._objective)
        cp._removed = None if self._removed is None else list(self._removed)
        return cp

    def objective(self) -> float:
        return float(self._objective(self))

    def get_context(self):
        return None


# -----------------------------
# Objective (closure, keeps EMA & history; always returns finite float)
# -----------------------------
def make_objective(instance, scen_small, cache, smooth_alpha: float):
    history: List[float] = []
    ema = {"val": None}

    def obj_core(state: VRPSDState) -> float:
        val, _ = evaluate_routes(instance, state.routes, cache, scen_small, None)
        v = float(val)
        if smooth_alpha and smooth_alpha > 0.0:
            ema["val"] = v if ema["val"] is None else (smooth_alpha * v + (1.0 - smooth_alpha) * ema["val"])
            v = float(ema["val"])
        history.append(v)
        return v

    def safe_objective(state: VRPSDState) -> float:
        try:
            v = obj_core(state)
            if v is None or not np.isfinite(v):
                return float(1e15)
            return float(v)
        except Exception:
            return float(1e15)

    return safe_objective, history


# -----------------------------
# Operators (ALNS 7.0 signatures)
#   destroy: op(state, rng, **kw) -> state   (must tag _removed)
#   repair:  op(state, rng, **kw) -> state   (must read state._removed)
# -----------------------------
def build_operators(alns: ALNS, distances: np.ndarray, py_rng: random.Random,
                    instance: 'VRPSDInstance', objective_fn):
    try:
        from .operators import (
            random_removal, worst_removal, shaw_removal,
            removed_customers, greedy_insert, regret_k_insert
        )
    except ImportError:
        from operators import (
            random_removal, worst_removal, shaw_removal,
            removed_customers, greedy_insert, regret_k_insert
        )

    def enforce_K(routes):
        return limit_routes_to_K(instance, routes, instance.K)

    def _flatten(sol):
        return [c for r in sol for c in r]

    def _removed_between(before_routes, after_routes):
        B = _flatten(before_routes)
        A = _flatten(after_routes)
        return [c for c in B if c not in A]

    # ---- destroy operators ----
    def destroy_random(state: VRPSDState, _rng, **_kw):
        new_routes = random_removal(
            state.routes,
            max(1, int(0.1 * sum(len(r) for r in state.routes))),
            py_rng
        )
        new_routes = enforce_K(new_routes)
        new_state = VRPSDState(new_routes, state._objective)
        new_state._removed = _removed_between(state.routes, new_routes)
        return new_state

    def destroy_worst(state: VRPSDState, _rng, **_kw):
        new_routes = worst_removal(
            state.routes,
            max(1, int(0.1 * sum(len(r) for r in state.routes))),
            distances,
            py_rng
        )
        new_routes = enforce_K(new_routes)
        new_state = VRPSDState(new_routes, state._objective)
        new_state._removed = _removed_between(state.routes, new_routes)
        return new_state

    def destroy_shaw(state: VRPSDState, _rng, **_kw):
        new_routes = shaw_removal(
            state.routes,
            max(1, int(0.1 * sum(len(r) for r in state.routes))),
            distances,
            py_rng
        )
        new_routes = enforce_K(new_routes)
        new_state = VRPSDState(new_routes, state._objective)
        new_state._removed = _removed_between(state.routes, new_routes)
        return new_state

    alns.add_destroy_operator(destroy_random)
    alns.add_destroy_operator(destroy_worst)
    alns.add_destroy_operator(destroy_shaw)

    # ---- repair operators ----
    def _infer_removed(routes):
        full = set(range(1, instance.lam.shape[0]))
        present = set(_flatten(routes))
        return list(full - present)

    def repair_greedy(state: VRPSDState, _rng, **_kw):
        rem = state._removed if state._removed is not None else _infer_removed(state.routes)
        new_routes = greedy_insert(state.routes, rem, distances, py_rng)
        new_routes = enforce_K(new_routes)
        return VRPSDState(new_routes, state._objective)

    def repair_regret2(state: VRPSDState, _rng, **_kw):
        rem = state._removed if state._removed is not None else _infer_removed(state.routes)
        new_routes = regret_k_insert(state.routes, rem, distances, 2, py_rng)
        new_routes = enforce_K(new_routes)
        return VRPSDState(new_routes, state._objective)

    def repair_regret3(state: VRPSDState, _rng, **_kw):
        rem = state._removed if state._removed is not None else _infer_removed(state.routes)
        new_routes = regret_k_insert(state.routes, rem, distances, 3, py_rng)
        new_routes = enforce_K(new_routes)
        return VRPSDState(new_routes, state._objective)

    alns.add_repair_operator(repair_greedy)
    alns.add_repair_operator(repair_regret2)
    alns.add_repair_operator(repair_regret3)


# -----------------------------
# Config & result
# -----------------------------
@dataclass
class ALNSSolveConfig:
    seed: int = 42
    time_limit_sec: float = None           # wall time limit (seconds); None to disable
    iters: int = 2000
    start_temp: float = 200.0
    cool_rate: float = 0.9995              # exponential cooling step in (0,1)
    end_temp: float = 1e-3
    small_samples: int = 96
    large_samples: int = 2048
    num_starts: int = 3
    use_adaptive_selection: bool = True
    roulette_scores: List[float] = field(default_factory=lambda: [5.0, 2.0, 1.0, 0.2])
    roulette_decay: float = 0.85
    smooth_alpha: float = 0.1


@dataclass
class SolveResult:
    routes: 'Solution'
    small_cost: float
    large_cost: float
    history: List[float]


# -----------------------------
# Stop conditions
# -----------------------------
class WallTimeStop:
    def __init__(self, limit_sec: float):
        self.limit = float(limit_sec)
        self._start = None
    def start(self, *args, **kwargs):
        self._start = time.time()
    def reset(self, *args, **kwargs):
        self._start = time.time()
    def __call__(self, *args, **kwargs) -> bool:
        if self._start is None:
            self._start = time.time()
        return (time.time() - self._start) >= self.limit


def _build_selector(cfg: ALNSSolveConfig, np_rng: np.random.Generator):
    if cfg.use_adaptive_selection:
        return RouletteWheel(scores=cfg.roulette_scores,
                             decay=cfg.roulette_decay,
                             num_destroy=1, num_repair=1)
    else:
        return RandomSelect(np_rng)


# -----------------------------
# Solve
# -----------------------------
def solve_vrpsd_with_alns(instance: 'VRPSDInstance', config: 'ALNSSolveConfig') -> 'SolveResult':
    # Two RNGs: numpy for ALNS; python for our custom ops
    py_rng = random.Random(config.seed)
    np_rng = np.random.default_rng(config.seed)
    distances = instance.distance_matrix()

    customers = list(range(1, instance.lam.shape[0]))
    best_routes = None
    best_small = float("inf")
    best_hist: List[float] = []
    cache = SAACache()

    scen_small = ScenarioManager(instance.lam, seed=12345).sample(config.small_samples)
    scen_large = ScenarioManager(instance.lam, seed=54321).sample(config.large_samples)

    for _ in range(config.num_starts):
        py_rng.shuffle(customers)
        init_routes = greedy_split_by_capacity(instance, customers)
        init_routes = limit_routes_to_K(instance, init_routes, instance.K)

        # objective closure attached to state
        obj_fn, obj_hist = make_objective(instance, scen_small, cache, smooth_alpha=config.smooth_alpha)
        state = VRPSDState(init_routes, obj_fn)

        alns = ALNS(np_rng)
        build_operators(alns, distances, py_rng, instance, obj_fn)

        sa = SimulatedAnnealing(start_temperature=config.start_temp,
                                end_temperature=config.end_temp,
                                step=config.cool_rate,
                                method='exponential')

        # stop conditions
        if Or is not None:
            stops = [MaxIterations(config.iters), NoImprovement(200)]
            if config.time_limit_sec is not None:
                stops.append(WallTimeStop(config.time_limit_sec))
            stop = stops[0]
            for st in stops[1:]:
                stop = Or(stop, st)
        else:
            class OrStop:
                def __init__(self,*ss): self.ss=ss
                def __call__(self,*a,**k): return any(s(*a,**k) for s in self.ss)
                def start(self,*a,**k):
                    for s in self.ss: getattr(s,"start",lambda *x,**y:None)()
                def reset(self,*a,**k):
                    for s in self.ss: getattr(s,"reset",lambda *x,**y:None)()
            stops = [MaxIterations(config.iters), NoImprovement(200)]
            if config.time_limit_sec is not None:
                stops.append(WallTimeStop(config.time_limit_sec))
            stop = OrStop(*stops)

        select = _build_selector(config, np_rng)

        # ALNS 7.0 signature
        result: Result = alns.iterate(state, select, sa, stop)

        # evaluate best
        small, large = evaluate_routes(instance, result.best_state.routes, cache, scen_small, scen_large)
        if small < best_small:
            best_small = small
            best_routes = result.best_state.routes
            best_hist = list(obj_hist)

    _, large = evaluate_routes(instance, best_routes, cache, scen_small, scen_large)
    return SolveResult(routes=best_routes, small_cost=best_small, large_cost=float(large), history=best_hist)