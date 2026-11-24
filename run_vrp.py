import sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json

import numpy as np
import torch
import re
import urllib.request
from scipy.stats import ranksums

from ekt_gl import EKTGL, EKTGLConfig, CVRPTask
from ekt_gl.evolution import Population

A_NAMES = ["A-n32-k5","A-n34-k5","A-n36-k5","A-n38-k5","A-n44-k6",
           "A-n46-k7","A-n48-k7","A-n54-k7","A-n60-k9","A-n62-k8"]

# ---- số liệu paper (Table V cột EKT-GL / MTEA-AST, Table IX LKH & EKT-GL best-of-20) ----
PAPER_EKTGL = [9.01,8.60,9.31,8.23,10.69,11.23,12.64,14.45,16.28,15.18]
PAPER_MTEA  = [9.27,8.71,9.71,8.81,11.30,12.08,13.49,15.47,17.94,16.86]
PAPER_BEST  = [8.29,8.23,8.69,7.90,10.26,10.26,12.28,13.76,15.54,14.61]
LKH_OPT     = [8.08,7.94,8.50,7.60,9.66,9.52,10.95,12.41,14.10,13.01]

URLS = [
    "http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/{s}/{n}.vrp",
    "http://vrp.galgos.inf.puc-rio.br/media/com_vrp/instances/{s}/{n}.vrp",
]

def _download(name, path):
    last = None
    for u in URLS:
        try:
            urllib.request.urlretrieve(u.format(s=name[0], n=name), path)
            return
        except Exception as e:
            last = e
    raise RuntimeError(
        f"Không tải được {name}: {last}\n"
        f"Hãy tải tay file {name}.vrp từ CVRPLIB và bỏ vào thư mục vrp_data/")

def _parse_vrp(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    cap = float(re.search(r"CAPACITY\s*:\s*([\d.]+)", text).group(1))
    coords, demand, section = {}, {}, None
    for line in text.splitlines():
        line = line.strip()
        if not line or line == "EOF":
            continue
        if line.endswith("_SECTION"):
            section = line
            continue
        if ":" in line:          # dòng header như NAME : ...
            continue
        parts = line.split()
        if section == "NODE_COORD_SECTION":
            coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
        elif section == "DEMAND_SECTION":
            demand[int(parts[0])] = float(parts[1])
        elif section == "DEPOT_SECTION":
            section = None
    ids = sorted(coords)         # node 1 là depot
    xy = np.array([coords[i] for i in ids], float)
    d = np.array([demand[i] for i in ids], float)
    return xy, d, cap

def load_raw(names, data_dir="vrp_data"):
    Path(data_dir).mkdir(exist_ok=True)
    raw = []
    for n in names:
        p = Path(data_dir) / f"{n}.vrp"
        if not p.exists():
            _download(n, str(p))
        xy, d, cap = _parse_vrp(p)
        raw.append((n, xy, d, cap))
    return raw

def build_tasks(raw, aux=False):
    rng = np.random.default_rng(1234)          # cố định bộ task phụ cho mọi seed
    tasks = []
    for n, xy, d, cap in raw:
        tasks.append(CVRPTask(n, xy, d, cap))
        if aux:
            for r in (0.1, 0.3, 0.5):
                cust = np.arange(1, len(xy))
                keep = np.sort(rng.choice(cust, int(round(len(cust)*(1-r))), replace=False))
                idx = np.concatenate([[0], keep])      # luôn giữ depot
                tasks.append(CVRPTask(f"{n}-del{int(r*100)}", xy[idx], d[idx], cap))
    return tasks

def _run_impl(args):
    seed, gens, aux, mode = args
    torch.set_num_threads(1)
    torch.manual_seed(seed)                    # solver chỉ seed numpy, cần seed thêm torch
    tasks = build_tasks(load_raw(A_NAMES), aux)
    cfg = EKTGLConfig(generations=gens)
    if mode == "ektgl":
        res = EKTGL(cfg).solve(tasks, seed=seed)
        return {n: v.cost for n, v in res.best.items()}
    # STO: cùng toán tử tiến hoá, KHÔNG transfer
    rng = np.random.default_rng(seed)
    pops = [Population(t, cfg.population_size, rng) for t in tasks]
    for _ in range(gens):
        for p in pops:
            p.evolve(cfg.mutation_rate, cfg.local_search_passes)
    return {t.name: p.best.cost for t, p in zip(tasks, pops)}

def run_one(args):
    seed, gens, aux, mode = args
    f = Path("results") / f"{mode}_{int(aux)}_{gens}_{seed}.json"
    if f.exists():
        return json.loads(f.read_text())
    out = _run_impl(args)
    f.parent.mkdir(exist_ok=True)
    f.write_text(json.dumps(out))
    return out

def collect(mode, runs, gens, aux, workers):
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(run_one, [(s, gens, aux, mode) for s in range(runs)]))
    print(f"[{mode}] {time.time()-t0:.0f}s tổng")
    return out

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    load_raw(A_NAMES) 
    runs  = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    gens  = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    work  = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    we = int(sys.argv[3]) if len(sys.argv) > 3 else 4   # EKT-GL: tốn RAM
    ws = int(sys.argv[4]) if len(sys.argv) > 4 else 6   # STO: nhẹ RAM
    ekt = collect("ektgl", runs, gens, False, we)
    sto = collect("sto",   runs, gens, False, ws)
    import pickle; pickle.dump((ekt, sto), open("raw_vrp_a.pkl", "wb"))

    lines = [f"{'Task':10s} {'EKT-GL':>12s} {'paper':>7s} {'delta%':>7s} {'STO':>12s} "
             f"{'p(vs STO)':>10s} {'best':>6s} {'LKH':>6s} {'gap%':>6s}"]
    for i, n in enumerate(A_NAMES):
        a = np.array([r[n] for r in ekt]); b = np.array([r[n] for r in sto])
        p = ranksums(a, b).pvalue
        sym = "-" if (p < 0.05 and a.mean() < b.mean()) else ("+" if p < 0.05 else "~")
        lines.append(
            f"{n:10s} {a.mean():6.2f}+-{a.std():4.2f} {PAPER_EKTGL[i]:7.2f} "
            f"{(a.mean()-PAPER_EKTGL[i])/PAPER_EKTGL[i]*100:7.1f} "
            f"{b.mean():6.2f}+-{b.std():4.2f} {p:9.3f}{sym} "
            f"{a.min():6.2f} {LKH_OPT[i]:6.2f} {(a.min()-LKH_OPT[i])/LKH_OPT[i]*100:6.1f}")
    lines.append("Ky hieu: '-' EKT-GL tot hon STO (p<0.05), '~' khong khac biet, '+' STO tot hon")
    text = "\n".join(lines)
    Path("results_vrp_a.txt").write_text(text, encoding="utf-8")
    print(text.encode("ascii", "replace").decode())