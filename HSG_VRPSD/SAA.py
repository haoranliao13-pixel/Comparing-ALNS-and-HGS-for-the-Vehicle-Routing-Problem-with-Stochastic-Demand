# -*- coding: utf-8 -*-
"""
SAA.py — 外层 SAA(样本平均近似) + 返库补救 的评估模块

提供:
  - ScenarioManager: 固定随机种生成并缓存需求场景 (CRN)
  - recourse_cost_one_scenario: 单场景下, 超载→返库→再出发 的成本
  - expected_cost_SAA: 给定路线, 在一批场景上计算期望成本
  - normalize_routes: 路线规范化(旋转最小化 + 排序) 便于去重与缓存
  - SAACache: 评估缓存
  - evaluate_routes: 对候选解集合做两阶段评估的小工具
  - build_dist: 从坐标构建欧氏距离矩阵(可选)

约定:
  - coords: [(x0,y0),(x1,y1),...,(xN,yN)], 0 为仓库, 客户为 1..N
  - routes: List[List[int]]，每条路线只含客户索引(1..N)，默认首尾回仓
  - demand/scenario: List[float]，长度 N+1，demand[0]=0
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Dict

import math
import numpy as np

__all__ = [
    "ScenarioManager",
    "recourse_cost_one_scenario",
    "expected_cost_SAA",
    "normalize_routes",
    "SAACache",
    "evaluate_routes",
    "build_dist",
]


# ---------------- 距离与工具 ----------------

def build_dist(coords: List[Tuple[float, float]]) -> List[List[float]]:
    """从坐标构造欧氏距离矩阵。可选：若你用 HGS 的 distance_matrix 输入，这个很方便。"""
    n = len(coords)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            d = math.hypot(xi - xj, yi - yj)
            D[i][j] = D[j][i] = d
    return D


def _dist_euclid(coords: List[Tuple[float, float]], i: int, j: int) -> float:
    xi, yi = coords[i]
    xj, yj = coords[j]
    return math.hypot(xi - xj, yi - yj)


# ---------------- 返库补救 & SAA ----------------

def recourse_cost_one_scenario(
    routes: List[List[int]],
    coords: List[Tuple[float, float]],
    Q: float,
    demand_vec: List[float],
) -> float:
    """
    在一个需求场景下计算总路程/成本:
    规则: 若到达客户 c 时 load + d_c > Q，则在该客户处立即回仓、再出发，之后继续服务。
    路径默认 0->...->0 闭合。
    """
    total = 0.0
    for r in routes:
        cur, load = 0, 0.0
        for c in r:
            total += _dist_euclid(coords, cur, c)
            d = demand_vec[c]
            if load + d <= Q:
                load += d
            else:
                # 返库再来: c->0->c
                total += _dist_euclid(coords, c, 0)
                total += _dist_euclid(coords, 0, c)
                load = d
            cur = c
        total += _dist_euclid(coords, cur, 0)
    return total


def expected_cost_SAA(
    routes: List[List[int]],
    coords: List[Tuple[float, float]],
    Q: float,
    scenarios: List[List[float]],
    return_samples: bool = False,
) -> float | Tuple[float, List[float]]:
    """
    在给定的一批场景上做样本平均期望成本。
    注意: 为确保对比公平，比较不同解时应该使用**同一批**场景(即 CRN)。
    """
    vals = [recourse_cost_one_scenario(routes, coords, Q, dem) for dem in scenarios]
    mean = float(sum(vals) / len(vals))
    return (mean, vals) if return_samples else mean


# ---------------- 路线规范化(去重/缓存友好) ----------------

def _canon_cycle(r: List[int]) -> Tuple[int, ...]:
    """把环状序列旋转到字典序最小的表示。"""
    if not r:
        return tuple()
    m = min(range(len(r)), key=lambda i: tuple(r[i:] + r[:i]))
    rr = r[m:] + r[:m]
    return tuple(rr)


def normalize_routes(routes: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
    """
    将一组路线规范化为:
      - 每条路线取字典序最小旋转表示
      - 再把所有路线按字典序排序
    返回可哈希的 tuple 作为键，用于去重/缓存
    """
    R = sorted((_canon_cycle(list(r)) for r in routes), key=lambda t: t if t else (-1,))
    return tuple(R)


# ---------------- 场景管理(共同随机数, CRN) ----------------

@dataclass
class ScenarioManager:
    """
    固定随机种生成一批泊松需求场景并缓存。比较不同解时复用同一批场景，能显著降低比较方差。
    lam: 客户 1..N 的泊松均值(不含仓库)，长度 N
    S:   场景数
    seed: 随机种
    """
    lam: List[float]
    S: int
    seed: int = 0

    def build(self) -> "ScenarioManager":
        rng = np.random.default_rng(self.seed)
        self._scenarios = [[0] + list(rng.poisson(self.lam)) for _ in range(int(self.S))]
        return self

    def get(self) -> List[List[float]]:
        return self._scenarios

    def regenerate(self, S: Optional[int] = None, seed: Optional[int] = None) -> "ScenarioManager":
        if S is not None:
            self.S = int(S)
        if seed is not None:
            self.seed = int(seed)
        return self.build()


# ---------------- 评估缓存与两阶段评估 ----------------

class SAACache:
    """SAA 评估缓存: key=normalize_routes(routes)，value=期望成本(在某批场景上)"""
    def __init__(self) -> None:
        self.memo: Dict[Tuple[Tuple[int, ...], ...], float] = {}

    def get(self, key) -> Optional[float]:
        return self.memo.get(key)

    def put(self, key, val: float) -> None:
        self.memo[key] = float(val)

    def clear(self) -> None:
        self.memo.clear()


def evaluate_routes(
    candidates: List[List[List[int]]],
    coords: List[Tuple[float, float]],
    Q: float,
    scenarios_small: List[List[float]],
    scenarios_big: Optional[List[List[float]]] = None,
    elite_frac: float = 1/3,
    min_elite: int = 5,
    cache: Optional[SAACache] = None,
) -> Tuple[Tuple[float, List[List[int]]], List[Tuple[float, List[List[int]]]], List[Tuple[float, List[List[int]]]]]:
    """
    对一组候选解做两阶段评估:
      - 阶段1: 用小样本 (scenarios_small) 评估所有候选
      - 阶段2: 对前 E 个精英用大样本 (scenarios_big) 精评(若提供)
    返回:
      - best_big: (cost, routes) 在大样本下的最优 (若无 big, 则为小样本最优)
      - scored_small: [(cost_small, routes)]  小样本评估结果(升序)
      - scored_big:   [(cost_big,   routes)]  大样本评估结果(升序，若无 big 则为空列表)
    """
    # 阶段1: 小样本评估并排序
    scored_small: List[Tuple[float, List[List[int]]]] = []
    for R in candidates:
        key = normalize_routes(R)
        if cache:
            val = cache.get(key)
        else:
            val = None
        if val is None:
            val = expected_cost_SAA(R, coords, Q, scenarios_small)
            if cache:
                cache.put(key, val)
        scored_small.append((val, R))
    scored_small.sort(key=lambda x: x[0])

    # 若没有大样本, 直接返回小样本的最优
    if scenarios_big is None or len(scenarios_big) == 0:
        best_small = scored_small[0]
        return best_small, scored_small, []

    # 阶段2: 大样本精评
    E = max(min_elite, int(math.ceil(len(scored_small) * elite_frac)))
    elites = scored_small[:E]

    rescored_big: List[Tuple[float, List[List[int]]]] = []
    for c_small, R in elites:
        c_big = expected_cost_SAA(R, coords, Q, scenarios_big)
        rescored_big.append((c_big, R))
    rescored_big.sort(key=lambda x: x[0])

    best_big = rescored_big[0]
    return best_big, scored_small, rescored_big
