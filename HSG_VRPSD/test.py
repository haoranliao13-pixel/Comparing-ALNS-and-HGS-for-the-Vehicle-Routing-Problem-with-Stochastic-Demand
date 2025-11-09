from __future__ import annotations
import os, sys, csv, math, importlib.util
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# ================== 参数（按需修改） ==================
SEEDS = 8            # HGS 候选解数量（不同随机种）
TIME_LIMIT = 10     # HGS 每次求解的时间(秒)
Q = 60             # 车辆容量
K = 35                # 车辆上界（传给 HGS；留空则由 HGS 自处理）
S_SMALL = 80        # 小样本场景数（共同随机数）
S_BIG = 300         # 大样本场景数（共同随机数；0 表示不进行复评）
ELITE_FRAC = 1/3     # 进入大样本复评的比例
MIN_ELITE = 5        # 进入大样本复评的最少候选数
SEED_SMALL = 1234    # 共同随机数：小样本 seed
SEED_BIG = 5678      # 共同随机数：大样本 seed
TIME_WALL = 10
HGS_CANDIDATES = [
    "pyhygese.py"
]
SAA_CANDIDATES = [
    "SAA.py"
]
# =====================================================
BASEDIR = os.path.dirname(os.path.realpath(__file__))
NODES_CSV  = r"C:\Users\haora\PyCharmMiscProject\datasets\nodes_400_centered.csv"
DEMAND_CSV = r"C:\Users\haora\PyCharmMiscProject\datasets\demand_400_centered.csv"


def _import_from_file_strict(pyname: str):
    path = os.path.join(BASEDIR, pyname)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"必须存在模块文件: {pyname}")
    mod_name = os.path.splitext(os.path.basename(pyname))[0]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {pyname}")
    mod = importlib.util.module_from_spec(spec)
    # 关键：预注册到 sys.modules，兼容 dataclass 在装饰时解析 __module__
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def _import_first_or_raise(candidates):
    last_err = None
    for fname in candidates:
        try:
            return _import_from_file_strict(fname)
        except Exception as e:
            last_err = e
    raise last_err if last_err else ImportError("未找到可用模块")

# 1) 严格导入 HGS 与 SAA
HGS = _import_first_or_raise(HGS_CANDIDATES)
SAA = _import_first_or_raise(SAA_CANDIDATES)

if not hasattr(HGS, "solve_cvrp_pyhygese"):
    raise AttributeError("HGS 模块缺少 solve_cvrp_pyhygese 接口。请在你的 HGS 封装中提供该函数。")

required_saa = ["ScenarioManager", "evaluate_routes", "SAACache"]
for name in required_saa:
    if not hasattr(SAA, name):
        raise AttributeError(f"SAA 模块缺少 {name} 接口。请在你的 SAA 脚本中提供。")

# 2) 读入数据
def read_nodes_and_demands(nodes_csv: str, demand_csv: str):
    coords = {}
    with open(nodes_csv, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#") or row[0].lower() == "node_id":
                continue
            idx = int(row[0]); x = float(row[1]); y = float(row[2])
            coords[idx] = (x, y)
    n = max(coords.keys())
    coords_list = [coords[i] for i in range(n+1)]  # 0..n

    lam = [0.0]*(n+1)
    with open(demand_csv, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#") or row[0].lower() == "node_id":
                continue
            idx = int(row[0]); l = float(row[1])
            lam[idx] = l
    lam[0] = 0.0
    return coords_list, lam


def build_distance(coords: List[Tuple[float,float]]):
    n = len(coords)
    D = [[0.0]*n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i+1, n):
            xj, yj = coords[j]
            d = math.hypot(xi-xj, yi-yj)
            D[i][j] = D[j][i] = d
    return D

coords, lam_vec = read_nodes_and_demands(NODES_CSV, DEMAND_CSV)
D = build_distance(coords)
demand_det = lam_vec[:]  # 期望需求；已确保 demand_det[0]=0

# === 可行性自检 + K 下限提示/自动修正（放在第一次 HGS 调用之前）===
lam = np.asarray(demand_det, dtype=float)  # 0号是仓库
max_lam = float(lam[1:].max())
sum_lam = float(lam[1:].sum())
K_min   = int(np.ceil(sum_lam / Q))
print(f"[check] max λ={max_lam:.3f}, sum λ={sum_lam:.3f}, ceil(sum/Q)={K_min}, Q={Q}, K={K}")

if max_lam > Q:
    raise ValueError(f"不可行：存在客户期望需求 {max_lam:.3f} > Q={Q}")

# 只提示：把下面两行留一行你喜欢的策略
if K is not None and K < K_min:
    print(f"[warn] 你给的 K={K} < 下限 {K_min}，可能不可行；建议设为 ≥ {K_min} 或 K=None")
# 或者自动修正：
# if K is None or K < K_min:
#     print(f"[auto] 调整 K: {K} -> {K_min}")
#     K = K_min


# 3) 共同随机数场景（使用你的 SAA 接口）
sm_small = SAA.ScenarioManager(lam=list(map(float, lam_vec[1:])), S=int(S_SMALL), seed=int(SEED_SMALL)).build()
scenarios_small = sm_small.get()

scenarios_big = []
if S_BIG and S_BIG > 0:
    sm_big = SAA.ScenarioManager(lam=list(map(float, lam_vec[1:])), S=int(S_BIG), seed=int(SEED_BIG)).build()
    scenarios_big = sm_big.get()

import time
start_time = time.time()
deadline = start_time + float(TIME_WALL)

def time_left():
    return max(0.0, deadline - time.time())

def time_up():
    return time.time() >= deadline

# 4) 用 HGS 在确定性需求上生成候选路线（多随机种，受总时间墙约束）
candidates: List[List[List[int]]] = []
for s in range(int(SEEDS)):
    if time_up():
        print(f"[timewall] 时间已到，停止继续跑 HGS（已收集 {len(candidates)} 个候选）")
        break

    # 给单次 HGS 的时间不超过全局剩余时间（留一点缓冲），至少 0.2 秒
    per_call_limit = min(float(TIME_LIMIT), max(0.2, time_left() * 0.85))
    if per_call_limit < 0.25:
        print(f"[timewall] 剩余时间不足以再跑一个 HGS 实例（left={time_left():.2f}s），跳出。")
        break

    routes_i, _ = HGS.solve_cvrp_pyhygese(
        dist_matrix=D,
        demand=demand_det,   # 长度 = N+1, 第0位=0
        Q=float(Q),
        K=int(K) if K is not None else None,
        time_limit=float(per_call_limit),   # <<< 不超过剩余时间
        seed=int(s),
        verbose=False,
    )
    # 转换为仅客户索引（去除 0）
    R = []
    for path in routes_i:
        r = [int(x) for x in path if int(x) != 0]
        R.append(r)
    candidates.append(R)

# 若一个候选都没有，兜底：至少跑一次“极短”HGS
if not candidates and not time_up():
    per_call_limit = max(0.2, min(float(TIME_LIMIT), time_left() * 0.85))
    routes_i, _ = HGS.solve_cvrp_pyhygese(
        dist_matrix=D,
        demand=demand_det,
        Q=float(Q),
        K=int(K) if K is not None else None,
        time_limit=float(per_call_limit),
        seed=0,
        verbose=False,
    )
    R = []
    for path in routes_i:
        r = [int(x) for x in path if int(x) != 0]
        R.append(r)
    if R:
        candidates.append(R)


# 5) 用 SAA 评估候选解并选最优（若时间紧张，跳过大样本复评）
use_big = bool(scenarios_big) and (time_left() > 5.0)   # 剩余>5秒才做大样本
if not candidates:
    raise RuntimeError("[timewall] 未得到任何候选路线，无法评估。")

best, scored_small, rescored_big = SAA.evaluate_routes(
    candidates=candidates,
    coords=coords,
    Q=float(Q),
    scenarios_small=scenarios_small,
    scenarios_big=scenarios_big if use_big else None,
    elite_frac=float(ELITE_FRAC),
    min_elite=int(MIN_ELITE),
    cache=SAA.SAACache(),
)
best_cost, best_routes = best

stop_reason = "time_wall" if time_up() else ("normal" if use_big else "no_big_due_to_time")
print(f"[Solver] stop_reason={stop_reason}, candidates={len(candidates)}, cost={best_cost:.4f}, left={time_left():.2f}s")


# 6) 打印与可视化
print("=== 小样本评估(升序 Top5) ===")
for c, R in scored_small[:5]:
    print(f"{c:.4f}\t{R}")
if rescored_big:
    print("=== 大样本精评(升序 Top5) ===")
    for c, R in rescored_big[:5]:
        print(f"{c:.4f}\t{R}")
print("\n>>> 选中的方案 期望成本(返库补救)：{:.4f}".format(best_cost))
print(">>> 方案路线：", best_routes)


# —— 全局一点点美化（不依赖 seaborn）——
from typing import Iterable, Tuple, Optional
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (7.5, 5.2),
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.25,
    "font.size": 11,
})

def plot_history_unified(
    history: Iterable[Tuple[int, float]],
    *,
    algo_name: str = "ALNS",
    title: Optional[str] = None,
    show_global_best: bool = True,
    save_path: Optional[str] = None,
):
    """
    history: [(epoch, best_obj), ...]
    - 不在曲线末尾做点标注
    - 标题内展示 cost（全程最优 或 最后一条）
    """
    history = list(history)
    if not history:
        raise ValueError("history 为空")

    xs = [e for e, _ in history]
    ys = [v for _, v in history]
    last_best = ys[-1]
    global_best = min(ys)

    if title is None:
        cost = global_best if show_global_best else last_best
        title = f"{algo_name} convergence  |  cost = {cost:.2f}"

    fig, ax = plt.subplots()
    ax.plot(xs, ys, linewidth=1.8)
    ax.set_xlabel("Iteration (epoch)")
    ax.set_ylabel("Best objective")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()

from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

def _xy_from_coords(coords):
    try:
        xs = [float(v) for v in coords[:, 0]]
        ys = [float(v) for v in coords[:, 1]]
    except Exception:
        xs = [float(x) for x, _ in coords]
        ys = [float(y) for _, y in coords]
    return xs, ys

def plot_routes_unified(
    coords: List[Tuple[float, float]] | "np.ndarray",
    routes: List[List[int]],
    *,
    cost: Optional[float] = None,       # 可选：在标题显示
    algo_name: str = "VRPSD",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    annotate_routes: bool = True,
):
    """
    routes: [[...], [...], ...]  每条只含客户索引（不含 0）；coords[0] 是仓库
    """
    xs, ys = _xy_from_coords(coords)

    if title is None:
        title = f"Best routes ({algo_name})" + (f"  |  E[cost] = {cost:.2f}" if cost is not None else "")

    fig, ax = plt.subplots()
    ax.scatter(xs[1:], ys[1:], s=22, label="customers")
    ax.scatter([xs[0]], [ys[0]], marker="s", s=100, label="depot")

    for ridx, r in enumerate(routes, 1):
        seq = [0] + r + [0]
        px = [xs[i] for i in seq]; py = [ys[i] for i in seq]
        ax.plot(px, py, linewidth=1.8, label=f"R{ridx}")
        if annotate_routes and r:
            mid = r[len(r)//2]
            ax.text(xs[mid] + 0.15, ys[mid] + 0.15, f"R{ridx}", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8, ncols=2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()

# 若有历史就画；没有可以先不画
# plot_history_unified(history, algo_name="HGS+SAA", show_global_best=True,
#                      save_path="hgs_saa_convergence.png")

plot_routes_unified(coords, best_routes,
                    cost=best_cost, algo_name="HGS+SAA",
                    save_path="hgs_saa_routes.png")



