import argparse
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def sfloat(x):
    try:
        return float(x)
    except Exception:
        return None

def _edges_iter(MG):
    # Uniformiza iteração para Graph/DiGraph/Multi*
    if isinstance(MG, (nx.MultiGraph, nx.MultiDiGraph)):
        for u, v, k, data in MG.edges(keys=True, data=True):
            yield u, v, k, data
    else:
        for u, v, data in MG.edges(data=True):
            yield u, v, None, data

def get_weight(data, fallback="score"):
    """
    Retorna o peso:
      - se existir 'weight', usa;
      - senão, fallback:
          'score' -> tanh(score/10)
          'unit'  -> 1.0
          'zero'  -> 0.0
          'skip'  -> None (sinaliza para pular)
    """
    w = sfloat(data.get("weight")) if ("weight" in data) else None
    if w is not None:
        return w
    if fallback == "score":
        sc = sfloat(data.get("score"))
        if sc is not None:
            return math.tanh(sc / 10.0)
        return None
    if fallback == "unit":
        return 1.0
    if fallback == "zero":
        return 0.0
    # 'skip'
    return None

def build_bins(args, values):
    if args.bins is not None:
        # nº de bins entre -1 e 1
        return np.linspace(-1.0, 1.0, args.bins + 1)
    # step
    step = args.step
    # protege contra step estranho
    if step <= 0:
        step = 0.1
    # garante que 1.0 está incluído exatamente
    edges = np.arange(-1.0, 1.0 + step, step)
    edges[-1] = 1.0
    return edges

def plot_hist(binedges, counts, outpng, title="Histograma de pesos das arestas", xlabel="Peso (w)"):
    centers = 0.5 * (binedges[:-1] + binedges[1:])
    widths = (binedges[1:] - binedges[:-1])
    plt.figure(figsize=(10,5), dpi=150)
    plt.bar(centers, counts, width=widths, align="center")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Contagem de arestas")
    # xticks menos poluídos
    try:
        step = binedges[1] - binedges[0]
        tick_every = max(1, int(round(0.2 / step)))  # tenta ~5 marcas no eixo
        sel = slice(0, len(centers), tick_every)
        xt = centers[sel]
        plt.xticks(xt, [f"{x:.1f}" for x in xt], rotation=0)
    except Exception:
        pass
    plt.tight_layout()
    plt.savefig(outpng)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Histograma de pesos de arestas em [-1,1] a partir de um .gexf")
    ap.add_argument("--gexf", required=True, help="Caminho para o arquivo .gexf")
    ap.add_argument("--outdir", default="auto", help="Pasta de saída (default: <stem>.gexf__weights)")
    ap.add_argument("--step", type=float, default=0.1, help="Tamanho do bin (ignorado se --bins for usado)")
    ap.add_argument("--bins", type=int, default=None, help="Número de bins entre -1 e 1 (ignora --step)")
    ap.add_argument("--fallback", choices=["score", "unit", "skip", "zero"], default="score",
                    help="Fallback quando a aresta não tem 'weight'")
    args = ap.parse_args()

    gexf_path = Path(args.gexf)
    stem = gexf_path.stem
    outdir = Path(args.outdir if args.outdir != "auto" else f"{stem}.gexf__weights")
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Ler grafo (mantém tipo: Graph/DiGraph/Multi*)
    G = nx.read_gexf(gexf_path)

    # 2) Extrair pesos
    rows = []
    weights = []
    missing = 0
    for u, v, k, data in _edges_iter(G):
        w = get_weight(data, fallback=args.fallback)
        if w is None:
            missing += 1
            continue
        rows.append({"source": u, "target": v, "key": k, "weight": w})
        weights.append(w)

    # 3) Salvar lista de arestas
    df_edges = pd.DataFrame(rows)
    df_edges.to_csv(outdir / "weights.csv", index=False)

    if len(weights) == 0:
        # nada a plotar
        summary = {
            "num_edges_read": G.number_of_edges(),
            "num_with_weight_or_fallback": 0,
            "num_missing_skipped": missing,
            "note": "Nenhuma aresta com peso (ou após fallback escolhido)."
        }
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Nenhuma aresta com peso disponível para histograma. Veja summary.json.")
        return

    w = np.array(weights, dtype=float)

    # 4) Estatísticas e contagens de sinal
    summary = {
        "num_edges_read": G.number_of_edges(),
        "num_with_weight_or_fallback": int(w.size),
        "num_missing_skipped": int(missing),
        "min": float(np.min(w)),
        "max": float(np.max(w)),
        "mean": float(np.mean(w)),
        "median": float(np.median(w)),
        "std": float(np.std(w, ddof=0)),
        "count_neg": int(np.sum(w < 0)),
        "count_zero": int(np.sum(w == 0)),
        "count_pos": int(np.sum(w > 0)),
    }

    # 5) Bins e histograma em [-1,1]
    bins = build_bins(args, w)
    counts, edges = np.histogram(w, bins=bins)

    # contabiliza fora do range
    summary["count_below_-1"] = int(np.sum(w < -1.0))
    summary["count_above_1"]  = int(np.sum(w > 1.0))
    summary["count_in_range"] = int(w.size - summary["count_below_-1"] - summary["count_above_1"])

    # 6) Salvar hist CSV
    df_hist = pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "count": counts
    })
    df_hist.to_csv(outdir / "weight_hist.csv", index=False)

    # 7) Plot PNG
    plot_hist(edges, counts, outdir / "weight_hist.png",
              title=f"Histograma de pesos das arestas ({stem})", xlabel="Peso (w)")

    # 8) Summary JSON
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Resultados salvos em:", outdir.resolve())
    print(" - weights.csv")
    print(" - weight_hist.csv")
    print(" - weight_hist.png")
    print(" - summary.json")

if __name__ == "__main__":
    main()

