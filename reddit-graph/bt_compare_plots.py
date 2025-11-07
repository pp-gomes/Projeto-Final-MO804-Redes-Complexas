#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bt_compare_plots.py — Comparativos (observado vs nulo) USANDO APENAS o contexto 'gexf'.

Ele lê os CSVs produzidos por bt_null_model.py (na pasta <stem>.gexf__null por padrão):
  gexf_triads_observed.csv
  gexf_null_hist_absBT.csv
  gexf_null_hist_BT.csv
  gexf_null_summary_vs_observed.csv

Cria uma pasta de saída automaticamente (default: <stem>.gexf__compare) e salva:
  - gexf_compare_absBT.png, gexf_compare_BT.png
  - gexf_compare_mean_absBT.png, gexf_compare_pctforte_bal.png, gexf_compare_pctforte_desbal.png, etc.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_edges_from_null_hist(df_hist: pd.DataFrame):
    left  = df_hist["bin_left"].to_numpy()
    right = df_hist["bin_right"].to_numpy()
    edges = np.concatenate([left[:1], right])
    centers = 0.5 * (left + right)
    return edges, centers


def observed_hist_from_edges(values: np.ndarray, edges: np.ndarray):
    counts, _ = np.histogram(values, bins=edges)
    return counts.astype(float)


def plot_hist_compare(centers, null_mean, null_std, obs_counts, title, xlabel, outpath: Path):
    plt.figure(figsize=(8,5), dpi=150)
    upper = null_mean + null_std
    lower = np.maximum(null_mean - null_std, 0.0)
    plt.fill_between(centers, lower, upper, alpha=0.25, label="Nulo (média ± sd)")
    plt.plot(centers, null_mean, linewidth=1.5, label="Nulo (média)")
    plt.step(centers, obs_counts, where="mid", linewidth=1.5, label="Observado")
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("Contagem de tríades")
    plt.legend(); plt.tight_layout(); plt.savefig(outpath); plt.close()


def plot_metric_bars(metric: str, obs_val, null_mean, null_std, outpath: Path, perc=False):
    labels = ["gexf"]; x = np.arange(len(labels)); width = 0.35
    y_obs  = np.array([obs_val], dtype=float)
    y_null = np.array([null_mean], dtype=float)
    y_err  = np.array([null_std], dtype=float)
    if perc:
        y_obs *= 100; y_null *= 100; y_err *= 100; ylab = "Valor (%)"
    else:
        ylab = "Valor"
    plt.figure(figsize=(6,5), dpi=150)
    plt.bar(x - width/2, y_obs,  width=width, label="Observado")
    plt.bar(x + width/2, y_null, width=width, yerr=y_err, capsize=4, label="Nulo (média ± sd)")
    plt.xticks(x, labels); plt.ylabel(ylab); plt.title(f"Comparativo — {metric}")
    plt.legend(); plt.tight_layout(); plt.savefig(outpath); plt.close()


def parse_args():
    ap = argparse.ArgumentParser(description="Comparativos (observado vs nulo) somente para .gexf.")
    ap.add_argument("--gexf", required=True, help="Caminho do .gexf (para inferir pastas)")
    ap.add_argument("--null-dir", default="auto", help="Pasta com CSVs do null (default: <stem>.gexf__null)")
    ap.add_argument("--outdir", default="auto", help="Pasta de saída (default: <stem>.gexf__compare)")
    return ap.parse_args()


def main():
    args = parse_args()
    stem = Path(args.gexf).stem
    null_dir = Path(args.null_dir if args.null_dir != "auto" else f"{stem}.gexf__null")
    outdir   = Path(args.outdir   if args.outdir   != "auto" else f"{stem}.gexf__compare")
    outdir.mkdir(parents=True, exist_ok=True)

    # Carregar arquivos do nulo/observado
    df_obs  = pd.read_csv(null_dir / "gexf_triads_observed.csv")      # BT, absBT
    df_nabs = pd.read_csv(null_dir / "gexf_null_hist_absBT.csv")      # bin_left, bin_right, count_mean, count_std
    df_nbt  = pd.read_csv(null_dir / "gexf_null_hist_BT.csv")
    df_sum  = pd.read_csv(null_dir / "gexf_null_summary_vs_observed.csv").set_index("metric")

    # Histogramas comparativos
    edges_abs, centers_abs = load_edges_from_null_hist(df_nabs)
    obs_counts_abs = observed_hist_from_edges(df_obs["absBT"].to_numpy(), edges_abs)
    plot_hist_compare(centers_abs, df_nabs["count_mean"].to_numpy(), df_nabs["count_std"].to_numpy(),
                      obs_counts_abs, "gexf: |BT| — Observado vs Nulo", "|BT|",
                      outdir / "gexf_compare_absBT.png")

    edges_bt, centers_bt = load_edges_from_null_hist(df_nbt)
    obs_counts_bt = observed_hist_from_edges(df_obs["BT"].to_numpy(), edges_bt)
    plot_hist_compare(centers_bt, df_nbt["count_mean"].to_numpy(), df_nbt["count_std"].to_numpy(),
                      obs_counts_bt, "gexf: BT — Observado vs Nulo", "BT",
                      outdir / "gexf_compare_BT.png")

    # Barras (algumas métricas)
    metrics = ["mean_absBT", "%forte_bal", "%forte_desbal", "mean_BT", "median_absBT"]
    for m in metrics:
        metric_name = m if (m in df_sum.index) else ("median_BT" if m=="median_absBT" and "median_BT" in df_sum.index else None)
        if not metric_name: continue
        obs_val   = float(df_sum.loc[metric_name, "obs"])
        null_mean = float(df_sum.loc[metric_name, "null_mean"])
        null_std  = float(df_sum.loc[metric_name, "null_std"])
        perc = metric_name.startswith("%")
        plot_metric_bars(metric_name, obs_val, null_mean, null_std,
                         outdir / f"gexf_compare_{metric_name.replace('%','pct')}.png", perc=perc)

    print("Figuras salvas em:", outdir.resolve())


if __name__ == "__main__":
    import numpy as np
    main()
