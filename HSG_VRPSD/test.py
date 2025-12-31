from __future__ import annotations
import os, sys, csv, math, importlib.util
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

SEEDS = 8           
TIME_LIMIT = 10     
Q = 220            
K = 90               
S_SMALL = 80        
S_BIG = 300         
ELITE_FRAC = 1/3    
MIN_ELITE = 5        
SEED_SMALL = 1234    
SEED_BIG = 5678     
TIME_WALL = 10
HGS_CANDIDATES = [
    "pyhygese.py"
]
SAA_CANDIDATES = [
    "SAA.py"
]
# =====================================================
BASEDIR = os.path.dirname(os.path.realpath(__file__))
NODES_CSV  = r"C:\Users\haora\PyCharmMiscProject\CVRPLib\csv\nodes_X-n303-k21.csv"
DEMAND_CSV = r"C:\Users\haora\PyCharmMiscProject\CVRPLib\csv\demand_X-n303-k21.csv"


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
    """
    Read node coordinates and stochastic demand parameters from CSV files.

    Compatible with:
      - with header:  id,x,y   /  id,lambda (or id,demand)
      - without header
      - UTF-8 with BOM
    """
    def _is_int_token(s: str) -> bool:
        s = s.strip()
        if not s:
            return False
        if s[0] in "+-":
            s = s[1:]
        return s.isdigit()

    # ---- nodes ----
    coords = {}
    with open(nodes_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            c0 = row[0].strip()
            if not c0 or c0.startswith("#"):
                continue
            c0l = c0.lower()
            # skip headers like: id,x,y or node_id,x,y
            if c0l in ("id", "node_id"):
                continue
            # skip any non-numeric first token (robust to accidental headers)
            if not _is_int_token(c0):
                continue
            if len(row) < 3:
                continue
            idx = int(c0)
            x = float(row[1]); y = float(row[2])
            coords[idx] = (x, y)

    if not coords:
        raise RuntimeError(f"No coordinates parsed from {nodes_csv}")

    n = max(coords.keys())
    # ensure 0..n exist (your downstream assumes this)
    missing = [i for i in range(n + 1) if i not in coords]
    if missing:
        raise RuntimeError(f"Missing node ids in {nodes_csv}: {missing[:10]}{'...' if len(missing)>10 else ''}")

    coords_list = [coords[i] for i in range(n + 1)]  # 0..n

    # ---- demands (lambda) ----
    lam = [0.0] * (n + 1)
    with open(demand_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            c0 = row[0].strip()
            if not c0 or c0.startswith("#"):
                continue
            c0l = c0.lower()
            # skip headers like: id,lambda  / id,demand / node_id,lambda
            if c0l in ("id", "node_id"):
                continue
            if not _is_int_token(c0):
                continue
            if len(row) < 2:
                continue
            idx = int(c0)
            l = float(row[1])
            if 0 <= idx <= n:
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
demand_det = lam_vec[:]

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


# === Build true iteration-like expected costs in generation order (small-sample only) ===
ecost_iter_small = []
try:
    if candidates:
        # Use a cache so identical routes don't get re-evaluated
        _tmp_cache = SAA.SAACache()
        for _R in candidates:
            # Evaluate this single candidate on the small-sample CRN
            _best1, _sc_small1, _ = SAA.evaluate_routes(
                candidates=[_R],
                coords=coords,
                Q=float(Q),
                scenarios_small=scenarios_small,
                scenarios_big=None,
                elite_frac=1.0,
                min_elite=1,
                cache=_tmp_cache,
            )
            _cost1 = float(_sc_small1[0][0])
            ecost_iter_small.append(_cost1)
except Exception as e:
    print(f"[warn] 构造真实小样本迭代成本序列失败: {e}")

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
from typing import Iterable, Tuple, Optional, Sequence, List
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

# ====== Academic plotting style (no seaborn) ======
def set_academic_mpl_style(*, base_fontsize: int = 11) -> None:
    """
    Set a clean, publication-friendly Matplotlib style that is:
    - serif font (Computer Modern-like)
    - thin grids, visible minor ticks
    - vector-friendly defaults
    """
    mpl.rcParams.update({
        "figure.figsize": (8.5, 6.0),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.transparent": True,
        "font.size": base_fontsize,
        "axes.labelsize": base_fontsize,
        "axes.titlesize": base_fontsize + 1,
        "legend.fontsize": base_fontsize - 1,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "axes.grid": True,
        "grid.color": "#999999",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "lines.markersize": 4.5,
        "text.usetex": False,                 # keep portable by default
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.borderpad": 0.2,
        "axes.formatter.useoffset": False,
        "axes.formatter.limits": (-3, 4),
    })

# Apply style once
set_academic_mpl_style()

def _save(fig, path: Optional[str]):
    if not path:
        return
    ext = os.path.splitext(path)[1].lower()
    # Encourage vector formats for papers
    if ext in (".pdf", ".svg"):
        fig.savefig(path, bbox_inches="tight")
    else:
        fig.savefig(path, dpi=300, bbox_inches="tight")

def _xy_from_coords(coords):
    try:
        xs = [float(v) for v in coords[:, 0]]
        ys = [float(v) for v in coords[:, 1]]
    except Exception:
        xs = [float(x) for x, _ in coords]
        ys = [float(y) for _, y in coords]
    return xs, ys

def plot_history_unified(
    history: Iterable[Tuple[int, float]],
    *,
    algo_name: str = "ALNS",
    title: Optional[str] = None,
    show_global_best: bool = True,
    save_path: Optional[str] = None,
    with_running_best: bool = True,
    logy: bool = False,
):
    """
    history: [(iteration, best_obj_so_far), ...]

    Enhancements:
      - cleaner style
      - optional running-best overlay
      - optional log-scale on y-axis
      - vector-friendly save (pdf/svg) when extension suggests it
    """
    history = list(history)
    if not history:
        raise ValueError("history 为空")

    xs = [int(e) for e, _ in history]
    ys = [float(v) for _, v in history]

    global_best = float(np.min(ys))
    last_best   = float(ys[-1])

    if title is None:
        cost = global_best if show_global_best else last_best
        title = f"Convergence of {algo_name}  |  cost = {cost:.2f}"

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(xs, ys, marker="o", linestyle="-", label="best so far")

    if with_running_best:
        running = np.minimum.accumulate(ys)
        ax.plot(xs, running, linestyle="--", label="running best")

    if logy:
        ax.set_yscale("log")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.minorticks_on()
    ax.legend(ncols=2)

    _save(fig, save_path)
    plt.show()

def plot_expected_cost_iteration(
    costs: Sequence[float] | Sequence[Tuple[int, float]],
    *,
    algo_name: str = "HGS+SAA",
    title: Optional[str] = "Expected cost (small-sample) over iterations",
    save_path: Optional[str] = None,
    with_running_best: bool = False,
    logy: bool = False,
):
    """
    Draw "Expected cost (small-sample) over iterations" like the reference style.
    Accepts either:
        - costs = [c0, c1, c2, ...]                (implicit iterations 1..n)
        - costs = [(iter0, c0), (iter1, c1), ...]  (explicit iterations)
    """
    if len(costs) == 0:
        raise ValueError("costs 为空")

    if isinstance(costs[0], (tuple, list)) and len(costs[0]) >= 2:  # type: ignore[index]
        xs = [int(it) for it, _ in costs]      # type: ignore[assignment]
        ys = [float(c) for _, c in costs]      # type: ignore[assignment]
    else:
        xs = list(range(1, len(costs) + 1))    # type: ignore[arg-type]
        ys = [float(c) for c in costs]         # type: ignore[arg-type]

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(xs, ys, marker="o", linestyle="-")

    if with_running_best:
        running = np.minimum.accumulate(ys)
        ax.plot(xs, running, linestyle="--")

    if logy:
        ax.set_yscale("log")

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Expected cost (SAA small sample)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.minorticks_on()

    _save(fig, save_path)
    plt.show()

def plot_routes_unified(
    coords: List[Tuple[float, float]] | "np.ndarray",
    routes: List[List[int]],
    *,
    cost: Optional[float] = None,
    algo_name: str = "VRPSD",
    title: Optional[str] = "Final routes",
    save_path: Optional[str] = None,
    annotate_routes: bool = False,
):
    """
    Draw routes in the style of the reference:
    - Blue square depot
    - Orange circular customer nodes
    - Colored route polylines with small circular markers
    - Clean grid, equal aspect
    """
    xs, ys = _xy_from_coords(coords)

    fig, ax = plt.subplots(constrained_layout=True)

    # Plot routes with color cycle; small markers along the lines
    for r in routes:
        seq = [0] + r + [0]
        px = [xs[i] for i in seq]; py = [ys[i] for i in seq]
        ax.plot(px, py, marker="o", linewidth=1.6)

    # Customers (orange) and depot (blue square)
    ax.scatter(xs[1:], ys[1:], s=28, label="Customers")
    ax.scatter([xs[0]], [ys[0]], marker="s", s=120, label="Depot")

    if annotate_routes and routes:
        for ridx, r in enumerate(routes, 1):
            if r:
                mid = r[len(r)//2]
                ax.text(xs[mid], ys[mid], f"R{ridx}", fontsize=9, ha="center", va="center")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=True, fancybox=False, edgecolor="0.8", loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    ax.minorticks_on()

    _save(fig, save_path)
    plt.show()

# === Example call (kept) ===
plot_routes_unified(coords, best_routes,
                    cost=best_cost, algo_name="HGS+SAA",
                    save_path="hgs_saa_routes.pdf")

# === Auto-plot expected cost curve (true iteration order) ===
try:
    if 'ecost_iter_small' in globals() and ecost_iter_small:
        _xs = list(range(1, len(ecost_iter_small)+1))
        _ys = [float(v) for v in ecost_iter_small]
        # Prefer the main plotting function if present
        _f = globals().get('plot_expected_cost_iteration')
        if callable(_f):
            _f(_ys,
               title="Expected cost (small-sample) over iterations",
               save_path="expected_cost_small_sample.pdf",
               with_running_best=False)
        else:
            # Fallback minimal plot to guarantee a figure is saved
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(constrained_layout=True)
            ax.plot(_xs, _ys, marker="o", linestyle="-")
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Expected cost (SAA small sample)")
            ax.set_title("Expected cost (small-sample) over iterations")
            ax.grid(True, which="both", alpha=0.3)
            ax.minorticks_on()
            fig.savefig("expected_cost_small_sample.pdf", bbox_inches="tight")
            plt.show()
except Exception as e:
    print(f"[warn] 绘制 expected cost 迭代曲线失败: {e}")
