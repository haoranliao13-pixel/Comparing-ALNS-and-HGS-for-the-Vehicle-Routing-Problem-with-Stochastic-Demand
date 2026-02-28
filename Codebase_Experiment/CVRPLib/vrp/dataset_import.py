
import re
from pathlib import Path
import requests
import pandas as pd

NAMES = [
    "X-n101-k25",
    "X-n200-k36",
    "X-n303-k21",
    "X-n401-k29",
    "X-n502-k39",
    "X-n599-k92",
    "X-n701-k44",
    "X-n801-k40",
    "X-n895-k37",
    "X-n1001-k43",
]


VRP_DIR = Path("vrp")
CSV_DIR = Path("csv")
VRP_DIR.mkdir(exist_ok=True)
CSV_DIR.mkdir(exist_ok=True)

BASES = [
    "https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/",
    "https://vrp.galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/",
    "http://vrp.galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X/",
]

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36"
}

def download_vrp(name: str) -> Path:
    out = VRP_DIR / f"{name}.vrp"
    if out.exists():
        return out

    last = None
    for base in BASES:
        url = f"{base}{name}.vrp"
        try:
            r = requests.get(url, headers=UA, timeout=60, allow_redirects=True)
            if r.status_code == 200 and b"NODE_COORD_SECTION" in r.content:
                out.write_bytes(r.content)
                return out
            last = f"{url} -> {r.status_code}"
        except Exception as e:
            last = f"{url} -> {repr(e)}"
    raise RuntimeError(f"Failed to download {name}. Last: {last}")

def parse_sections(vrp_text: str):
    m1 = re.search(r"NODE_COORD_SECTION\s*(.*?)\s*DEMAND_SECTION", vrp_text, re.S)
    m2 = re.search(r"DEMAND_SECTION\s*(.*?)\s*DEPOT_SECTION", vrp_text, re.S)
    if not m1 or not m2:
        raise ValueError("Cannot find NODE_COORD_SECTION / DEMAND_SECTION")

    coords_lines = [ln.strip() for ln in m1.group(1).splitlines() if ln.strip()]
    dem_lines = [ln.strip() for ln in m2.group(1).splitlines() if ln.strip()]

    coords = [(int(a[0]), float(a[1]), float(a[2])) for a in (ln.split() for ln in coords_lines)]
    demands = [(int(a[0]), float(a[1])) for a in (ln.split() for ln in dem_lines)]

    df_nodes = pd.DataFrame(coords, columns=["old_id", "x", "y"])
    df_dem = pd.DataFrame(demands, columns=["old_id", "demand"])
    return df_nodes, df_dem

def reindex_depot0(df_nodes: pd.DataFrame, df_dem: pd.DataFrame, depot_old_id: int = 1):
    old_ids = df_nodes["old_id"].tolist()
    if depot_old_id not in old_ids:
        raise ValueError(f"Depot id {depot_old_id} not found (expected 1).")

    customers = [i for i in old_ids if i != depot_old_id]
    mapping = {depot_old_id: 0}
    mapping.update({old: new for new, old in enumerate(customers, start=1)})

    nodes = df_nodes.copy()
    dem = df_dem.copy()
    nodes["id"] = nodes["old_id"].map(mapping)
    dem["id"] = dem["old_id"].map(mapping)

    nodes = nodes.sort_values("id")[["id", "x", "y"]].reset_index(drop=True)
    dem = dem.sort_values("id")[["id", "demand"]].reset_index(drop=True)
    return nodes, dem

def export_csv(name: str, nodes: pd.DataFrame, dem: pd.DataFrame):
    # VRPSD input: lambda = deterministic demand (mean-demand construction)
    lam = dem.rename(columns={"demand": "lambda"}).copy()
    lam.loc[lam["id"] == 0, "lambda"] = 0.0

    # Your current test.py reader assumes NO header (first row must be numeric),
    # so we write the default files WITHOUT headers.
    nodes.to_csv(CSV_DIR / f"nodes_{name}.csv", index=False, header=False)
    lam.to_csv(CSV_DIR / f"demand_{name}.csv", index=False, header=False)

    # Also keep headered copies for convenience (optional).
    nodes.to_csv(CSV_DIR / f"nodes_{name}.csv", index=False, header=True)
    lam.to_csv(CSV_DIR / f"demand_{name}.csv", index=False, header=True)

# --- run ---
for nm in NAMES:
    vrp_path = download_vrp(nm)
    txt = vrp_path.read_text(encoding="utf-8", errors="ignore")
    df_nodes, df_dem = parse_sections(txt)
    nodes, dem = reindex_depot0(df_nodes, df_dem, depot_old_id=1)
    export_csv(nm, nodes, dem)
    print("OK:", nm, "->", f"csv/nodes_{nm}.csv", f"csv/demand_{nm}.csv")
