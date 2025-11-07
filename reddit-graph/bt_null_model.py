#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bt_null_model.py — Modelo nulo de BT/|BT| por randomização de pesos (APENAS .gexf).
Cria pasta de saída automaticamente (default: <stem>.gexf__null).

Modos de embaralhamento:
- global      : embaralha todos os pesos globalmente.
- per_source  : embaralha apenas entre arestas SAINDO do mesmo nó.
- signs_only  : mantém magnitudes e embaralha só os sinais.
"""

import math, argparse, random
from itertools import combinations
from pathlib import Path
from typing import Dict, Tuple, List, Iterable

import numpy as np
import pandas as pd
import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def classify_bt_signed(bt: float) -> str:
    if bt > 0.5:
        return "fortemente balanceada"
    if bt > 0.0:
        return "levemente balanceada"
    if bt > -0.5:
        return "levemente desbalanceada"
    return "fortemente desbalanceada"


def sfloat(x):
    try:
        return float(x)
    except Exception:
        return None


def normalize_series(vals: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
    if not vals:
        return vals
    absmax = max(abs(v) for v in vals.values())
    if absmax == 0:
        return {k: 0.0 for k in vals}
    if absmax <= 1.0:
        return dict(vals)
    return {k: v / absmax for k, v in vals.items()}


def _edges_iterator(MG: nx.Graph):
    if isinstance(MG, (nx.MultiDiGraph, nx.MultiGraph)):
        for u, v, _k, data in MG.edges(keys=True, data=True):
            yield u, v, data
    else:
        for u, v, data in MG.edges(data=True):
            yield u, v, data


def load_gexf_graph(gexf_path: Path) -> nx.DiGraph:
    MG = nx.read_gexf(gexf_path)
    tmp = {}
    for u, v, data in _edges_iterator(MG):
        w = None
        if "weight" in data:
            w = sfloat(data.get("weight"))
        if w is None:
            sc = sfloat(data.get("score"))
            if sc is not None:
                w = math.tanh(sc / 10.0)
        if w is None:
            w = 0.0
        tmp.setdefault((u, v), []).append(w)
    weights = {k: sum(v) / len(v) for k, v in tmp.items()}
    weights = normalize_series(weights)
    G = nx.DiGraph()
    for (a, b), w in weights.items():
        G.add_edge(a, b, weight=float(w))
    return G


def list_ego_triads_edges(G: nx.DiGraph) -> List[Tuple[Tuple[str,str], Tuple[str,str], Tuple[str,str]]]:
    triads = []
    for a in G.nodes():
        succ = list(G.successors(a))
        if len(succ) < 2:
            continue
        for b, c in combinations(succ, 2):
            if not (G.has_edge(a, b) and G.has_edge(a, c)):
                continue
            if G.has_edge(b, c):
                triads.append(((a, b), (a, c), (b, c)))
            if G.has_edge(c, b):
                triads.append(((a, b), (a, c), (c, b)))
    return triads


def compute_bt_from_edges(weight_map: Dict[Tuple[str,str], float],
                          triads: List[Tuple[Tuple[str,str],Tuple[str,str],Tuple[str,str]]]) -> pd.DataFrame:
    rows = []
    for (ab, ac, bc) in triads:
        wab = weight_map.get(ab); wac = weight_map.get(ac); wbc = weight_map.get(bc)
        if wab is None or wac is None or wbc is None:
            continue
        bt = float(wab) * float(wac) * float(wbc)
        rows.append({"edge_ab": ab, "edge_ac": ac, "edge_bc": bc, "BT": bt, "absBT": abs(bt)})
    return pd.DataFrame(rows)


def extract_weight_map(G: nx.DiGraph) -> Dict[Tuple[str,str], float]:
    return {(u, v): float(d.get("weight", 0.0)) for u, v, d in G.edges(data=True)}


def shuffle_weights_global(edges: List[Tuple[str,str]], weights: List[float], rng: random.Random):
    w = weights[:]; rng.shuffle(w)
    return {e: w[i] for i, e in enumerate(edges)}


def shuffle_weights_per_source(edges_by_src: Dict[str, List[Tuple[str,str]]],
                               weights_by_src: Dict[str, List[float]],
                               rng: random.Random):
    out = {}
    for u, elist in edges_by_src.items():
        w = weights_by_src[u][:]; rng.shuffle(w)
        for i, e in enumerate(elist):
            out[e] = w[i]
    return out


def shuffle_signs_only(edges: List[Tuple[str,str]], weights: List[float], rng: random.Random):
    mags  = [abs(w) for w in weights]
    signs = [0.0 if w == 0 else (1.0 if w > 0 else -1.0) for w in weights]
    rng.shuffle(signs)
    return {edges[i]: mags[i] * signs[i] for i in range(len(edges))}


def summarize_stats(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {k: (0 if k=="triads" else float("nan")) for k in
                ["triads","mean_BT","median_BT","mean_absBT","median_absBT",
                 "%forte_bal","%leve_bal","%leve_desbal","%forte_desbal"]}
    triads = len(df)
    a = df["absBT"]
    cats = pd.Series([classify_bt_signed(x) for x in df["BT"]]).value_counts()
    f = lambda k: float(cats.get(k, 0) / triads)
    return {
        "triads": triads,
        "mean_BT": float(df["BT"].mean()),
        "median_BT": float(df["BT"].median()),
        "mean_absBT": float(a.mean()),
        "median_absBT": float(a.median()),
        "%forte_bal": f("fortemente balanceada"),
        "%leve_bal": f("levemente balanceada"),
        "%leve_desbal": f("levemente desbalanceada"),
        "%forte_desbal": f("fortemente desbalanceada"),
    }


def hist_series(values: np.ndarray, bins: int, range_tuple):
    counts, edges = np.histogram(values, bins=bins, range=range_tuple)
    return counts.astype(float), edges


def run_gexf(G: nx.DiGraph, n_iters: int, shuffle_mode: str, bins: int, seed: int, outdir: Path, prefix: str):
    rng = random.Random(seed)
    triads_edges = list_ego_triads_edges(G)
    weight_map_obs = extract_weight_map(G)
    df_obs = compute_bt_from_edges(weight_map_obs, triads_edges)
    obs_stats = summarize_stats(df_obs)

    pd.DataFrame([obs_stats]).to_csv(outdir / f"{prefix}_observed_stats.csv", index=False)
    df_obs.to_csv(outdir / f"{prefix}_triads_observed.csv", index=False)

    edges = list(weight_map_obs.keys())
    weights = [weight_map_obs[e] for e in edges]

    edges_by_src = {}
    weights_by_src = {}
    if shuffle_mode == "per_source":
        for u in G.nodes():
            out_edges = list(G.out_edges(u))
            if out_edges:
                edges_by_src[u] = out_edges
                weights_by_src[u] = [weight_map_obs[e] for e in out_edges]

    iter_rows, bt_hist_all, abs_hist_all = [], [], []
    for it in range(1, n_iters+1):
        if shuffle_mode == "global":
            wmap = shuffle_weights_global(edges, weights, rng)
        elif shuffle_mode == "per_source":
            wmap = shuffle_weights_per_source(edges_by_src, weights_by_src, rng)
        elif shuffle_mode == "signs_only":
            wmap = shuffle_signs_only(edges, weights, rng)
        else:
            raise ValueError(f"shuffle_mode inválido: {shuffle_mode}")

        df_it = compute_bt_from_edges(wmap, triads_edges)
        row = summarize_stats(df_it); row["iter"] = it
        iter_rows.append(row)

        if not df_it.empty:
            c_bt,   edges_bt   = hist_series(df_it["BT"].to_numpy(),    bins=bins, range_tuple=(-1.0, 1.0))
            c_abs,  edges_abs  = hist_series(df_it["absBT"].to_numpy(), bins=bins, range_tuple=( 0.0, 1.0))
        else:
            c_bt  = np.zeros(bins); c_abs = np.zeros(bins)
            edges_bt  = np.linspace(-1.0, 1.0, bins+1)
            edges_abs = np.linspace( 0.0, 1.0, bins+1)

        bt_hist_all.append(c_bt); abs_hist_all.append(c_abs)

    df_null_iters = pd.DataFrame(iter_rows)
    df_null_iters.to_csv(outdir / f"{prefix}_null_iter_stats.csv", index=False)

    bt_hist_all  = np.stack(bt_hist_all,  axis=0)
    abs_hist_all = np.stack(abs_hist_all, axis=0)

    df_bt_hist = pd.DataFrame({
        "bin_left":  edges_bt[:-1],
        "bin_right": edges_bt[1:],
        "count_mean": bt_hist_all.mean(axis=0),
        "count_std":  bt_hist_all.std(axis=0, ddof=1) if n_iters > 1 else np.zeros_like(bt_hist_all.mean(axis=0)),
    })
    df_abs_hist = pd.DataFrame({
        "bin_left":  edges_abs[:-1],
        "bin_right": edges_abs[1:],
        "count_mean": abs_hist_all.mean(axis=0),
        "count_std":  abs_hist_all.std(axis=0, ddof=1) if n_iters > 1 else np.zeros_like(abs_hist_all.mean(axis=0)),
    })
    df_bt_hist.to_csv(outdir / f"{prefix}_null_hist_BT.csv", index=False)
    df_abs_hist.to_csv(outdir / f"{prefix}_null_hist_absBT.csv", index=False)

    def mean_std(series):
        return float(series.mean()), (float(series.std(ddof=1)) if len(series) > 1 else float("nan"))

    metrics = ["mean_BT","median_BT","mean_absBT","median_absBT",
               "%forte_bal","%leve_bal","%leve_desbal","%forte_desbal"]
    rows = []
    for m in metrics:
        mu, sd = mean_std(df_null_iters[m].dropna())
        obs = obs_stats[m]
        z = (obs - mu) / sd if (sd and not np.isnan(sd)) else float("nan")
        rows.append({"metric": m, "obs": obs, "null_mean": mu, "null_std": sd, "zscore": z})
    pd.DataFrame(rows).to_csv(outdir / f"{prefix}_null_summary_vs_observed.csv", index=False)


def parse_args():
    ap = argparse.ArgumentParser(description="Modelo nulo de BT/|BT| (apenas .gexf).")
    ap.add_argument("--gexf", required=True, help="Caminho para o .gexf")
    ap.add_argument("--n-iters", type=int, default=100, help="Número de iterações do nulo")
    ap.add_argument("--shuffle-mode", choices=["global","per_source","signs_only"], default="global")
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="auto", help="Pasta de saída (default: <stem>.gexf__null)")
    return ap.parse_args()


def main():
    args = parse_args()
    stem = Path(args.gexf).stem
    outdir = Path(args.outdir if args.outdir != "auto" else f"{stem}.gexf__null")
    outdir.mkdir(parents=True, exist_ok=True)

    G = load_gexf_graph(Path(args.gexf))
    print(f"[gexf] |V|={G.number_of_nodes()} |E|={G.number_of_edges()}")
    run_gexf(G, args.n_iters, args.shuffle_mode, args.bins, args.seed, outdir, prefix="gexf")

    print("Arquivos salvos em:", outdir.resolve())


if __name__ == "__main__":
    main()
