#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bt_threshold_plus_signed.py — Análise de BT (tríades ego) com categorias ASSINADAS usando APENAS .gexf.
Cria uma pasta de saída automaticamente (default: <stem>.gexf__bt_signed).

Classificação assinada:
- BT > 0.5              -> fortemente balanceada
- 0 < BT <= 0.5         -> levemente balanceada
- -0.5 < BT <= 0        -> levemente desbalanceada
- BT <= -0.5            -> fortemente desbalanceada
"""

import math
import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, Tuple, Iterable

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


def _edges_iterator(MG: nx.Graph) -> Iterable[Tuple[str, str, dict]]:
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


def compute_bt_ego(G: nx.DiGraph) -> pd.DataFrame:
    rows = []
    for a in G.nodes():
        succ = list(G.successors(a))
        if len(succ) < 2:
            continue
        for b, c in combinations(succ, 2):
            if not (G.has_edge(a, b) and G.has_edge(a, c)):
                continue
            wab = float(G[a][b]["weight"]); wac = float(G[a][c]["weight"])

            if G.has_edge(b, c):
                wbc = float(G[b][c]["weight"])
                bt = wab * wac * wbc
                rows.append({"ego": a, "b": b, "c": c, "third_edge": "B->C",
                             "w_ab": wab, "w_ac": wac, "w_bc": wbc,
                             "BT": bt, "absBT": abs(bt),
                             "classe_signed": classify_bt_signed(bt)})
            if G.has_edge(c, b):
                wcb = float(G[c][b]["weight"])
                bt = wab * wac * wcb
                rows.append({"ego": a, "b": b, "c": c, "third_edge": "C->B",
                             "w_ab": wab, "w_ac": wac, "w_bc": wcb,
                             "BT": bt, "absBT": abs(bt),
                             "classe_signed": classify_bt_signed(bt)})
    return pd.DataFrame(rows)


def plot_histograms(df: pd.DataFrame, outdir: Path, tag: str, bins: int = 60):
    if df.empty:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,5), dpi=150)
    plt.hist(df["BT"], bins=bins); plt.title(f"Histograma de BT ({tag})")
    plt.xlabel("BT"); plt.ylabel("Frequência"); plt.tight_layout()
    plt.savefig(outdir / f"hist_BT_{tag}.png"); plt.close()

    plt.figure(figsize=(8,5), dpi=150)
    plt.hist(df["absBT"], bins=bins); plt.title(f"Histograma de |BT| ({tag})")
    plt.xlabel("|BT|"); plt.ylabel("Frequência"); plt.tight_layout()
    plt.savefig(outdir / f"hist_absBT_{tag}.png"); plt.close()


def export_filtered_gexf(G: nx.DiGraph, triads_df: pd.DataFrame, outpath: Path, abs_threshold: float):
    if triads_df.empty:
        nx.write_gexf(nx.DiGraph(), outpath); return outpath
    keep = set()
    for _, r in triads_df.iterrows():
        if r["absBT"] >= abs_threshold:
            a, b, c = r["ego"], r["b"], r["c"]
            keep.add((a, b)); keep.add((a, c))
            keep.add((b, c) if r["third_edge"] == "B->C" else (c, b))
    H = G.edge_subgraph(keep).copy()
    nx.write_gexf(H, outpath)
    return outpath


def parse_args():
    ap = argparse.ArgumentParser(description="BT assinado (apenas .gexf) + hist + GEXF filtrado por |BT|.")
    ap.add_argument("--gexf", required=True, help="Caminho para o .gexf")
    ap.add_argument("--abs-threshold", type=float, default=0.3, help="Limiar de |BT| para filtrar GEXF (default: 0.3)")
    ap.add_argument("--bins", type=int, default=60, help="Nº de bins dos histogramas")
    ap.add_argument("--outdir", default="auto", help="Pasta de saída (default: <stem>.gexf__bt_signed)")
    return ap.parse_args()


def main():
    args = parse_args()
    stem = Path(args.gexf).stem
    outdir = Path(args.outdir if args.outdir != "auto" else f"{stem}.gexf__bt_signed")
    outdir.mkdir(parents=True, exist_ok=True)

    G = load_gexf_graph(Path(args.gexf))
    df = compute_bt_ego(G)

    df.to_csv(outdir / "triads_gexf_signed.csv", index=False)

    summary = pd.DataFrame([{
        "contexto": "gexf",
        "triades": int(len(df)),
        "%forte_bal":  (df["classe_signed"] == "fortemente balanceada").mean() if len(df) else 0.0,
        "%leve_bal":   (df["classe_signed"] == "levemente balanceada").mean() if len(df) else 0.0,
        "%leve_desbal":(df["classe_signed"] == "levemente desbalanceada").mean() if len(df) else 0.0,
        "%forte_desbal":(df["classe_signed"] == "fortemente desbalanceada").mean() if len(df) else 0.0,
        "media_BT":  float(df["BT"].mean()) if len(df) else None,
        "mediana_BT":float(df["BT"].median()) if len(df) else None,
    }])
    summary.to_csv(outdir / "triads_summary_signed.csv", index=False)

    plot_histograms(df, outdir, "gexf", bins=args.bins)

    filtered_path = outdir / f"{stem}_bt_abs{args.abs_threshold}.gexf"
    export_filtered_gexf(G, df, filtered_path, args.abs_threshold)

    print(f"Saída: {outdir.resolve()}")
    print(" - triads_gexf_signed.csv, triads_summary_signed.csv")
    print(" - hist_BT_gexf.png, hist_absBT_gexf.png")
    print(f" - {filtered_path.name}")


if __name__ == "__main__":
    main()
