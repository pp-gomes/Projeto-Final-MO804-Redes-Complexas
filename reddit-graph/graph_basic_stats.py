#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
graph_basic_stats.py — Estatísticas básicas e distribuições a partir de um arquivo .gexf

O script:
1) Lê um .gexf (suporta Graph/DiGraph e Multi*).
2) Constrói também a versão não-direcionada simples (sem paralelas nem laços) para métricas que exigem conectividade.
3) Calcula e salva:
   - Nº de vértices e arestas; grau médio (total / in / out);
   - Coeficiente de clusterização GLOBAL (transitivity) + média de clusterização local;
   - Coeficiente de clusterização por nó + histograma (PNG);
   - Distância média (na maior componente do não-direcionado);
   - Diâmetro (exato para grafos até certo tamanho; senão, aproximação);
   - Densidade;
   - Distribuição de graus (bar chart, PNG);
   - Nº de componentes no não-direcionado + (se >1) distribuição dos tamanhos (PNG).
4) Salva CSVs/JSON e PNGs numa pasta de saída.

Uso:
  python graph_basic_stats.py --gexf "./politics.gexf"
Parâmetros úteis:
  --outdir auto         # pasta de saída (default: <stem>.gexf__basics)
  --bins 50             # bins dos histogramas (clusterização)
  --max-exact 5000      # no. máx de nós para cálculos exatos de distância/diâmetro
  --spl-samples 200     # nº de amostras para estimar distância média se o grafo for grande
  --seed 42             # semente para amostragem
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple, Iterable, List, Optional
import random
import math

import networkx as nx
import pandas as pd
import numpy as np

# matplotlib: 1 figura por gráfico, sem cores explícitas
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- utilidades de IO e conversão ----------

def sfloat(x):
    try:
        return float(x)
    except Exception:
        return None

def _edges_iterator(MG: nx.Graph) -> Iterable[Tuple[str, str, dict]]:
    """Iterador uniforme de arestas para Graph/DiGraph e MultiGraph/MultiDiGraph."""
    if isinstance(MG, (nx.MultiDiGraph, nx.MultiGraph)):
        for u, v, _k, data in MG.edges(keys=True, data=True):
            yield u, v, data
    else:
        for u, v, data in MG.edges(data=True):
            yield u, v, data

def read_gexf_simple(path: Path) -> nx.DiGraph | nx.Graph:
    """
    Lê GEXF e devolve um DiGraph/Graph simples (sem paralelas):
    - Se houver atributo 'weight', usa; senão tenta 'score' -> tanh(score/10); senão 1.0.
    - Para múltiplas arestas entre dois nós, mantém uma (ignora paralelas) — as métricas aqui são topológicas.
    """
    MG = nx.read_gexf(path)
    is_directed = MG.is_directed()
    G = nx.DiGraph() if is_directed else nx.Graph()

    for u, v, data in _edges_iterator(MG):
        if u == v:
            # Removemos laços para as métricas (evitam viés em grau/clusterização)
            continue
        w = None
        if "weight" in data:
            w = sfloat(data.get("weight"))
        if w is None and "score" in data:
            sc = sfloat(data.get("score"))
            if sc is not None:
                w = math.tanh(sc / 10.0)
        if w is None:
            w = 1.0
        # se já existe uma aresta, mantemos a primeira (topologia); peso não é crítico para métricas pedidas
        if not G.has_edge(u, v):
            G.add_edge(u, v, weight=float(w))
    return G

def to_undirected_simple(Gin: nx.Graph | nx.DiGraph) -> nx.Graph:
    """Converte para grafo simples não-direcionado (sem paralelas nem laços)."""
    if isinstance(Gin, nx.Graph) and not Gin.is_directed():
        G = Gin.copy()
    else:
        G = nx.Graph()
        G.add_nodes_from(Gin.nodes())
        for u, v in Gin.edges():
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)
    # remove laços se sobrar algum
    loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(loops)
    return G


# ---------- métricas que exigem conectividade ----------

def giant_component(G: nx.Graph) -> nx.Graph:
    """Maior componente conectada do grafo não-direcionado."""
    if G.number_of_nodes() == 0:
        return G.copy()
    comps = list(nx.connected_components(G))
    if not comps:
        return G.copy()
    gc_nodes = max(comps, key=len)
    return G.subgraph(gc_nodes).copy()

def average_shortest_path_length_safe(Gc: nx.Graph,
                                      max_exact: int = 5000,
                                      samples: int = 200,
                                      rng: Optional[random.Random] = None) -> float | None:
    """
    Se |V| <= max_exact: usa cálculo exato (nx.average_shortest_path_length).
    Caso contrário, estima com amostragem de fontes uniformes (BFS).
    """
    n = Gc.number_of_nodes()
    if n == 0 or n == 1:
        return 0.0
    if n <= max_exact:
        return nx.average_shortest_path_length(Gc)

    rng = rng or random.Random(42)
    nodes = list(Gc.nodes())
    k = min(samples, n)
    sources = rng.sample(nodes, k)
    total = 0.0
    count = 0
    for s in sources:
        lengths = nx.single_source_shortest_path_length(Gc, s)
        # soma distâncias até todos (exceto 0 para o próprio)
        total += sum(d for t, d in lengths.items() if t != s)
        count += (len(lengths) - 1)
    # média por fonte, depois média entre fontes
    return total / count if count > 0 else None

def diameter_safe(Gc: nx.Graph,
                  max_exact: int = 3000,
                  approx_sweeps: int = 20,
                  rng: Optional[random.Random] = None) -> int | None:
    """
    Diâmetro exato para grafos moderados; senão, aproximação por "double sweep" repetido.
    """
    n = Gc.number_of_nodes()
    if n == 0:
        return None
    if n == 1:
        return 0
    if n <= max_exact:
        try:
            return nx.diameter(Gc)
        except Exception:
            pass  # fallback para aproximação

    rng = rng or random.Random(42)
    nodes = list(Gc.nodes())
    best = 0
    # double-sweep: escolhe s, acha t mais distante; de t acha u mais distante; usa excêntricidades como lower bound
    for _ in range(approx_sweeps):
        s = rng.choice(nodes)
        dist_s = nx.single_source_shortest_path_length(Gc, s)
        if not dist_s:
            continue
        t = max(dist_s, key=dist_s.get)
        dist_t = nx.single_source_shortest_path_length(Gc, t)
        if dist_t:
            best = max(best, max(dist_t.values()))
    return int(best)


# ---------- plots ----------

def plot_hist(values, title, xlabel, outpng, bins=50):
    plt.figure(figsize=(8,5), dpi=150)
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.savefig(outpng)
    plt.close()

def plot_degree_bar(degree_counts: Dict[int, int], outpng: Path, title="Distribuição de graus"):
    if not degree_counts:
        return
    degs = sorted(degree_counts.keys())
    freqs = [degree_counts[d] for d in degs]
    plt.figure(figsize=(9,5), dpi=150)
    plt.bar(degs, freqs)
    plt.title(title)
    plt.xlabel("Grau")
    plt.ylabel("Número de nós")
    plt.tight_layout()
    plt.savefig(outpng)
    plt.close()


# ---------- main pipeline ----------

def main():
    ap = argparse.ArgumentParser(description="Estatísticas e distribuições básicas para um grafo .gexf")
    ap.add_argument("--gexf", required=True, help="Caminho para o arquivo .gexf")
    ap.add_argument("--outdir", default="auto", help="Pasta de saída (default: <stem>.gexf__basics)")
    ap.add_argument("--bins", type=int, default=50, help="Bins para histogramas (clusterização)")
    ap.add_argument("--max-exact", type=int, default=5000, help="Máximo de nós para cálculos exatos de dist/diâmetro")
    ap.add_argument("--spl-samples", type=int, default=200, help="Amostras para estimar dist. média em grafos grandes")
    ap.add_argument("--seed", type=int, default=42, help="Semente pseudoaleatória")
    args = ap.parse_args()

    gexf_path = Path(args.gexf)
    stem = gexf_path.stem
    outdir = Path(args.outdir if args.outdir != "auto" else f"{stem}.gexf__basics")
    outdir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    # 0) Carregar grafo simples (sem paralelas; remove laços)
    G = read_gexf_simple(gexf_path)
    directed = G.is_directed()

    # 1) Versão não-direcionada simples (para conectividade, clusterização etc.)
    GU = to_undirected_simple(G)

    # 2) Métricas globais básicas (no grafo original e no não-direcionado)
    n_dir = G.number_of_nodes()
    m_dir = G.number_of_edges()
    n_ud  = GU.number_of_nodes()
    m_ud  = GU.number_of_edges()

    # graus médios
    if directed:
        avg_in  = np.mean([d for _, d in G.in_degree()]) if n_dir > 0 else 0.0
        avg_out = np.mean([d for _, d in G.out_degree()]) if n_dir > 0 else 0.0
        avg_tot = np.mean([d for _, d in G.degree()]) if n_dir > 0 else 0.0
    else:
        avg_in = avg_out = None
        avg_tot = (2*m_dir / n_dir) if n_dir > 0 else 0.0

    # densidade (NetworkX já lida com direção)
    density_dir = nx.density(G)
    density_ud  = nx.density(GU)

    # 3) Clusterização (global e por nó) no não-direcionado
    try:
        transitivity = nx.transitivity(GU)  # global clustering (triangles/triads)
    except Exception:
        transitivity = None
    try:
        avg_clust = nx.average_clustering(GU)
    except Exception:
        avg_clust = None

    clust_by_node = nx.clustering(GU)  # dict node -> Cc
    # salvar CSV
    pd.DataFrame({"node": list(clust_by_node.keys()),
                  "clustering": list(clust_by_node.values())}) \
      .to_csv(outdir / "clustering_by_node.csv", index=False)
    # histograma de clusterização (distribuição)
    plot_hist(list(clust_by_node.values()),
              title="Distribuição do coeficiente de clusterização (nó a nó)",
              xlabel="Clustering local",
              outpng=outdir / "hist_clustering.png",
              bins=args.bins)

    # 4) Componentes (no não-direcionado)
    n_components = nx.number_connected_components(GU)
    comp_sizes = [len(c) for c in nx.connected_components(GU)]
    pd.DataFrame({"component_size": comp_sizes}).to_csv(outdir / "component_sizes.csv", index=False)
    if n_components > 1:
        # histograma (ou barra) de tamanhos das componentes
        size_counts = Counter(comp_sizes)
        # ordenar por tamanho crescente
        sizes_sorted = sorted(size_counts.keys())
        counts_sorted = [size_counts[s] for s in sizes_sorted]
        plt.figure(figsize=(9,5), dpi=150)
        plt.bar([str(s) for s in sizes_sorted], counts_sorted)
        plt.title("Distribuição dos tamanhos das componentes (não-direcionado)")
        plt.xlabel("Tamanho da componente")
        plt.ylabel("Quantidade de componentes")
        plt.tight_layout()
        plt.savefig(outdir / "component_size_distribution.png")
        plt.close()

    # 5) Distância média e diâmetro (maior componente não-direcionada)
    GU_gc = giant_component(GU)
    avg_path_len = average_shortest_path_length_safe(
        GU_gc, max_exact=args.max_exact, samples=args.spl_samples, rng=rng
    )
    diam = diameter_safe(GU_gc, max_exact=min(args.max_exact, 3000), approx_sweeps=20, rng=rng)

    # 6) Distribuição de graus (no não-direcionado)
    degs = [d for _, d in GU.degree()]
    deg_counts = Counter(degs)
    # CSV
    pd.DataFrame({"degree": list(deg_counts.keys()),
                  "count":  list(deg_counts.values())}) \
      .to_csv(outdir / "degree_distribution.csv", index=False)
    # gráfico (barras discretas)
    plot_degree_bar(deg_counts, outdir / "degree_distribution.png",
                    title="Distribuição de graus (não-direcionado)")

    # 7) Resumo final
    summary = {
        "directed": directed,
        "n_vertices_directed": n_dir,
        "n_arestas_directed": m_dir,
        "grau_medio_total_directed": avg_tot,
        "grau_medio_in_directed": avg_in,
        "grau_medio_out_directed": avg_out,
        "densidade_directed": density_dir,

        "n_vertices_undirected": n_ud,
        "n_arestas_undirected": m_ud,
        "grau_medio_undirected": (2*m_ud / n_ud) if n_ud > 0 else 0.0,
        "densidade_undirected": density_ud,

        "clusterizacao_global_transitivity": transitivity,
        "clusterizacao_media_local": avg_clust,

        "n_componentes_undirected": n_components,
        "tamanho_maior_componente": GU_gc.number_of_nodes(),

        "distancia_media_maior_componente": avg_path_len,
        "diametro_maior_componente": diam,
    }
    # salvar JSON e CSV
    import json
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(outdir / "summary.csv", index=False)

    # 8) Log bonito no terminal
    print(f"\nResultados salvos em: {outdir.resolve()}")
    print("Arquivos principais:")
    for fname in ["summary.json", "summary.csv",
                  "clustering_by_node.csv", "hist_clustering.png",
                  "degree_distribution.csv", "degree_distribution.png",
                  "component_sizes.csv"]:
        p = outdir / fname
        if p.exists():
            print(" -", p.name)
    if (outdir / "component_size_distribution.png").exists():
        print(" - component_size_distribution.png")
    print("\nResumo rápido:")
    print(f"  • Direcionado? {directed}")
    print(f"  • Nós/arestas (dir): {n_dir}/{m_dir} | densidade={density_dir:.6f} | grau_médio_total={avg_tot:.4f}"
          + (f" | in={avg_in:.4f} | out={avg_out:.4f}" if directed else ""))
    print(f"  • Nós/arestas (und): {n_ud}/{m_ud} | densidade={density_ud:.6f} | grau_médio={((2*m_ud/n_ud) if n_ud>0 else 0.0):.4f}")
    print(f"  • Clusterização: transitivity={transitivity}, média_local={avg_clust}")
    print(f"  • Componentes (und): {n_components} | maior componente: {GU_gc.number_of_nodes()} nós")
    print(f"  • Distância média (maior comp): {avg_path_len}")
    print(f"  • Diâmetro (maior comp): {diam}")


if __name__ == "__main__":
    main()
