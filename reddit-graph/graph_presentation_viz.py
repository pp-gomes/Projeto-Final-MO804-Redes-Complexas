#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
graph_presentation_viz.py — Visualização clara (estática + interativa) para apresentação de grafos .gexf

Mudanças pedidas:
  - Sem nomes/rótulos de nós (nem de arestas);
  - Arestas coloridas por sinal do peso: verde (w>0), vermelho (w<0), cinza-escuro (w=0);
  - Legenda explicando as cores das arestas (estático e interativo).

Saídas (em <stem>.gexf__viz/ por padrão):
  - viz_presentation.png / viz_presentation.svg         # figura estática
  - viz_presentation.html                                # figura interativa (Plotly), se disponível
  - node_metrics.csv (id, degree, community, x, y)      # métrica & layout por nó
  - edge_metrics.csv (source, target, weight)           # arestas finais
  - README_viz.txt                                       # parâmetros usados

Uso (exemplos):
  python graph_presentation_viz.py --gexf "./politics.gexf" --keep-gcc --interactive both
  python graph_presentation_viz.py --gexf "./politics.gexf" --keep-gcc --min-weight 0.2 --max-nodes 2000

Dependências:
  pip install networkx pandas numpy matplotlib
  (opcional, para HTML interativo) pip install plotly
  (opcional, melhor comunidades)   pip install python-louvain
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, Iterable, List, Optional
import math
import random
import numpy as np
import pandas as pd
import networkx as nx

# Matplotlib para saída estática
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines  # para a legenda das arestas


# ------------------------- utilidades de leitura/conversão -------------------------

def sfloat(x):
    try:
        return float(x)
    except Exception:
        return None

def _edges_iterator(MG: nx.Graph) -> Iterable[Tuple[str, str, dict]]:
    """Iterador uniforme para Graph/DiGraph e MultiGraph/MultiDiGraph."""
    if isinstance(MG, (nx.MultiDiGraph, nx.MultiGraph)):
        for u, v, _k, data in MG.edges(keys=True, data=True):
            yield u, v, data
    else:
        for u, v, data in MG.edges(data=True):
            yield u, v, data

def read_gexf_simple(path: Path) -> nx.DiGraph | nx.Graph:
    """
    Lê .gexf e devolve grafo simples (sem paralelas; remove laços).
    - Mantém direção se original for dirigido.
    - Peso preferido: 'weight'; fallback: tanh(score/10); senão 1.0.
    """
    MG = nx.read_gexf(path)
    is_directed = MG.is_directed()
    G = nx.DiGraph() if is_directed else nx.Graph()
    for u, v, data in _edges_iterator(MG):
        if u == v:
            continue  # remove laços
        w = None
        if "weight" in data:
            w = sfloat(data.get("weight"))
        if w is None and "score" in data:
            sc = sfloat(data.get("score"))
            if sc is not None:
                w = math.tanh(sc / 10.0)
        if w is None:
            w = 1.0
        if not G.has_edge(u, v):
            G.add_edge(u, v, weight=float(w))
    return G

def to_undirected_simple(G: nx.Graph | nx.DiGraph) -> nx.Graph:
    """Converte para grafo não-direcionado simples (sem paralelas nem laços) para layout/comunidades."""
    if isinstance(G, nx.Graph) and not G.is_directed():
        H = G.copy()
    else:
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        for u, v, data in G.edges(data=True):
            if u != v and not H.has_edge(u, v):
                H.add_edge(u, v, **{k: data[k] for k in data if k != "key"})
    H.remove_edges_from(list(nx.selfloop_edges(H)))
    return H

def keep_giant_component(G: nx.Graph) -> nx.Graph:
    """Mantém apenas a maior componente conectada (para GU)."""
    if G.number_of_nodes() == 0:
        return G.copy()
    comps = list(nx.connected_components(G))
    if not comps:
        return G.copy()
    gc_nodes = max(comps, key=len)
    return G.subgraph(gc_nodes).copy()


# ------------------------- comunidades, escalas e layout -------------------------

def detect_communities(GU: nx.Graph, seed: int = 42) -> Dict[str, int]:
    """
    Comunidades sobre o grafo NÃO-DIRECIONADO.
    Tenta Louvain (python-louvain); se não houver, usa greedy_modularity_communities do NetworkX.
    Retorna dict: node -> community_id (0..K-1).
    """
    try:
        import community as community_louvain  # python-louvain
        part = community_louvain.best_partition(GU, random_state=seed, weight=None)  # usa topologia
        uniq = {c: i for i, c in enumerate(sorted(set(part.values())))}
        return {n: uniq[c] for n, c in part.items()}
    except Exception:
        pass
    # Fallback: greedy modularity communities
    comms = list(nx.algorithms.community.greedy_modularity_communities(GU))
    mapping = {}
    for i, com in enumerate(comms):
        for n in com:
            mapping[n] = i
    for n in GU.nodes():
        if n not in mapping:
            mapping[n] = len(comms)
    return mapping

def scale_values(vals, min_size=6, max_size=28):
    """Escala valores positivos para [min_size, max_size]. Se todos iguais, usa tamanho médio."""
    v = np.array(list(vals), dtype=float)
    if len(v) == 0:
        return []
    v = np.maximum(v, 0.0)
    vmin, vmax = v.min(), v.max()
    if vmax - vmin < 1e-12:
        return [ (min_size + max_size)/2.0 ] * len(v)
    vv = (v - vmin) / (vmax - vmin)
    return list(min_size + vv * (max_size - min_size))

def coarse_layout(GU: nx.Graph, seed: int = 42) -> dict:
    """
    Layout rápido para grafos grandes (100% NetworkX):
      1) seleciona hubs (top ~40%, máx 3000);
      2) spring_layout leve no core;
      3) barycenter pros demais (ou jitter);
      4) refinamento leve fixando o core.
    """
    rng = random.Random(seed)
    n = GU.number_of_nodes()
    if n == 0:
        return {}
    core_size = min(3000, max(200, int(0.4 * n)))
    deg = dict(GU.degree())
    core_nodes = [u for u, _d in sorted(deg.items(), key=lambda kv: kv[1], reverse=True)[:core_size]]
    H = GU.subgraph(core_nodes).copy()

    k_core = 1.0 / math.sqrt(max(len(H), 1))
    try:
        pos_core = nx.spring_layout(H, seed=seed, k=k_core, iterations=200, threshold=1e-3, method="force")
    except TypeError:
        pos_core = nx.spring_layout(H, seed=seed, k=k_core, iterations=200, threshold=1e-3)
    except ValueError:
        pos_core = nx.spring_layout(H, seed=seed, k=k_core, iterations=200, threshold=1e-3)

    pos = dict(pos_core)
    for u in GU.nodes():
        if u in pos:
            continue
        neigh = [v for v in GU.neighbors(u) if v in pos]
        if neigh:
            xs = [pos[v][0] for v in neigh]; ys = [pos[v][1] for v in neigh]
            pos[u] = (sum(xs)/len(xs), sum(ys)/len(ys))
        else:
            pos[u] = (0.005 * (rng.random()-0.5), 0.005 * (rng.random()-0.5))

    k_all = 1.0 / math.sqrt(n)
    fixed = list(pos_core.keys())
    try:
        pos = nx.spring_layout(
            GU, seed=seed, k=k_all, iterations=(60 if n > 3000 else 120),
            threshold=1e-3, method="force", pos=pos, fixed=fixed
        )
    except TypeError:
        pos = nx.spring_layout(
            GU, seed=seed, k=k_all, iterations=(60 if n > 3000 else 120),
            threshold=1e-3, pos=pos, fixed=fixed
        )
    except ValueError:
        pos = nx.spring_layout(
            GU, seed=seed, k=k_all, iterations=(60 if n > 3000 else 120),
            threshold=1e-3, pos=pos, fixed=fixed
        )
    return pos

def build_layout(GU: nx.Graph, seed: int = 42):
    """
    Escolha automática e 100% em NetworkX:
      - <= 3000 nós: spring_layout (modo 'force');
      - > 3000 nós: coarse_layout.
    """
    n = max(GU.number_of_nodes(), 1)
    if n <= 3000:
        k = 1.0 / math.sqrt(n)
        try:
            return nx.spring_layout(GU, seed=seed, k=k, iterations=200, threshold=1e-3, method="force")
        except TypeError:
            return nx.spring_layout(GU, seed=seed, k=k, iterations=200, threshold=1e-3)
        except ValueError:
            return nx.spring_layout(GU, seed=seed, k=k, iterations=200, threshold=1e-3)
    return coarse_layout(GU, seed=seed)


# ------------------------- desenho estático e interativo -------------------------

def _split_edges_by_sign(Gf: nx.Graph | nx.DiGraph, GU: nx.Graph):
    """
    Separa listas de arestas por sinal do peso (w>0, w<0, w==0), restritas aos nós presentes em GU.
    Retorna (elist_pos, elist_neg, elist_zero).
    """
    pos_list, neg_list, zero_list = [], [], []
    for u, v, data in Gf.edges(data=True):
        if (u not in GU) or (v not in GU):
            continue
        w = sfloat(data.get("weight", 0.0))
        w = 0.0 if w is None else w
        if w > 0:
            pos_list.append((u, v))
        elif w < 0:
            neg_list.append((u, v))
        else:
            zero_list.append((u, v))
    return pos_list, neg_list, zero_list

def draw_static(Gf: nx.Graph | nx.DiGraph, GU: nx.Graph, pos, comm_map: Dict[str, int], deg_map: Dict[str, int],
                out_png: Path, out_svg: Path, title: str = ""):
    # Cores por comunidade (tab20 ciclando) — só nos NÓS
    coms = np.array([comm_map[n] for n in GU.nodes()])
    uniq = sorted(set(coms))
    cmap = plt.cm.get_cmap("tab20", max(len(uniq), 1))
    color_map = {c: cmap(i % cmap.N) for i, c in enumerate(uniq)}
    node_colors = [color_map[comm_map[n]] for n in GU.nodes()]

    # Tamanho por grau (não-direcionado)
    node_sizes = scale_values([deg_map[n] for n in GU.nodes()], min_size=10, max_size=40)

    # Arestas por sinal (usando pesos de Gf, mas desenhando nas posições de GU)
    elist_pos, elist_neg, elist_zero = _split_edges_by_sign(Gf, GU)

    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
    ax.set_axis_off()

    # Desenhar arestas por grupo (sem setas/rótulos)
    if elist_zero:
        nx.draw_networkx_edges(GU, pos, ax=ax, edgelist=elist_zero, edge_color="dimgray", width=0.6, alpha=0.45)
    if elist_pos:
        nx.draw_networkx_edges(GU, pos, ax=ax, edgelist=elist_pos, edge_color="green",   width=0.8, alpha=0.65)
    if elist_neg:
        nx.draw_networkx_edges(GU, pos, ax=ax, edgelist=elist_neg, edge_color="red",     width=0.8, alpha=0.65)

    # Nós (sem nomes)
    nx.draw_networkx_nodes(GU, pos, ax=ax, node_size=node_sizes, node_color=node_colors, linewidths=0.2, edgecolors="white")

    # Legenda das arestas
    legend_handles = [
        mlines.Line2D([], [], color="green",   label="aresta positiva (w > 0)", linewidth=2),
        mlines.Line2D([], [], color="red",     label="aresta negativa (w < 0)", linewidth=2),
        mlines.Line2D([], [], color="dimgray", label="aresta neutra (w = 0)",   linewidth=2),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)

    if title:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_svg)
    plt.close(fig)

def draw_interactive(Gf: nx.Graph | nx.DiGraph, GU: nx.Graph, pos, comm_map: Dict[str, int], deg_map: Dict[str, int],
                     out_html: Path, title: str = ""):
    """
    Interativo com Plotly:
      - 3 traces de arestas por sinal (cores iguais ao estático) com legendas;
      - nós sem nomes (hover sem id); cores por comunidade; tamanho por grau.
    """
    try:
        import plotly.graph_objs as go
        from plotly.offline import plot as plotly_save
    except Exception as e:
        print(f"[aviso] Plotly não está disponível ({e}). Pulando HTML interativo.")
        return

    # Build edge coordinate lists por grupo
    elist_pos, elist_neg, elist_zero = _split_edges_by_sign(Gf, GU)

    def edges_to_xy(edgelist):
        xs, ys = [], []
        for u, v in edgelist:
            x0, y0 = pos[u]; x1, y1 = pos[v]
            xs += [x0, x1, None]
            ys += [y0, y1, None]
        return xs, ys

    # Traces de arestas
    traces_edges = []
    if elist_pos:
        x, y = edges_to_xy(elist_pos)
        traces_edges.append(go.Scatter(
            x=x, y=y, mode="lines", line=dict(width=0.8, color="green"), opacity=0.65,
            hoverinfo="none", name="w > 0"
        ))
    if elist_neg:
        x, y = edges_to_xy(elist_neg)
        traces_edges.append(go.Scatter(
            x=x, y=y, mode="lines", line=dict(width=0.8, color="red"), opacity=0.65,
            hoverinfo="none", name="w < 0"
        ))
    if elist_zero:
        x, y = edges_to_xy(elist_zero)
        traces_edges.append(go.Scatter(
            x=x, y=y, mode="lines", line=dict(width=0.6, color="dimgray"), opacity=0.45,
            hoverinfo="none", name="w = 0"
        ))

    # Nodes (sem nomes nos hovers)
    nodes = list(GU.nodes())
    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    degrees = [deg_map[n] for n in nodes]
    sizes = scale_values(degrees, 10, 28)

    coms = [comm_map[n] for n in nodes]
    uniq = sorted(set(coms))
    cmap = plt.cm.get_cmap("tab20", max(len(uniq), 1))
    rgb = lambda rgba: f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},0.9)"
    colors = [rgb(cmap(uniq.index(c) % cmap.N)) for c in coms]

    # sem id no hover: apenas métricas
    hover_text = [f"grau: {deg_map[n]}<br>comunidade: {comm_map[n]}" for n in nodes]

    node_trace = go.Scatter(
        x=xs, y=ys,
        mode="markers",
        hoverinfo="text",
        text=hover_text,
        marker=dict(size=sizes, color=colors, line=dict(width=0.5, color="#ffffff")),
        name="nós",
        showlegend=False
    )

    layout = go.Layout(
        title=title or "",
        showlegend=True,  # <- legenda das arestas
        hovermode="closest",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.6)")
    )
    fig = go.Figure(data=[*traces_edges, node_trace], layout=layout)

    plotly_save(fig, filename=str(out_html), auto_open=False, include_plotlyjs="cdn")


# ------------------------- filtros práticos -------------------------

def maybe_filter_edges_by_weight(G: nx.Graph | nx.DiGraph, min_abs_weight: Optional[float]) -> nx.Graph | nx.DiGraph:
    if min_abs_weight is None:
        return G
    H = G.__class__()  # mantém direção
    H.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        w = sfloat(data.get("weight", 0.0))
        if w is None:
            w = 0.0
        if abs(w) >= min_abs_weight:
            H.add_edge(u, v, **data)
    return H

def maybe_limit_nodes(G: nx.Graph | nx.DiGraph, max_nodes: Optional[int]) -> nx.Graph | nx.DiGraph:
    if not max_nodes or G.number_of_nodes() <= max_nodes:
        return G
    # mantém os top-N por grau total (no grafo não-direcionado equivalente)
    GU = to_undirected_simple(G)
    deg = dict(GU.degree())
    top = sorted(deg.items(), key=lambda kv: kv[1], reverse=True)[:max_nodes]
    keep = set(n for n, _ in top)
    H = G.__class__()
    for n in G.nodes():
        if n in keep:
            H.add_node(n, **G.nodes[n])
    for u, v, data in G.edges(data=True):
        if u in keep and v in keep:
            H.add_edge(u, v, **data)
    return H


# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser(description="Visualização de apresentação para grafos .gexf (estático + interativo)")
    ap.add_argument("--gexf", required=True, help="Caminho do arquivo .gexf")
    ap.add_argument("--outdir", default="auto", help="Pasta de saída (default: <stem>.gexf__viz)")
    ap.add_argument("--keep-gcc", action="store_true", help="Manter apenas a maior componente (não-direcionado)")
    ap.add_argument("--min-weight", type=float, default=None, help="Filtrar arestas com |weight| >= limiar")
    ap.add_argument("--max-nodes", type=int, default=None, help="Se definido, mantém apenas os top-N nós por grau")
    ap.add_argument("--label-top-k", type=int, default=0, help="(Ignorado) — rótulos desabilitados por solicitação")
    ap.add_argument("--seed", type=int, default=42, help="Semente para layout/comunidades")
    ap.add_argument("--interactive", choices=["both", "html", "none"], default="both",
                    help="Gerar HTML interativo (plotly) e/ou apenas estático (png/svg)")
    args = ap.parse_args()

    gexf_path = Path(args.gexf)
    stem = gexf_path.stem
    outdir = Path(args.outdir if args.outdir != "auto" else f"{stem}.gexf__viz")
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Carregar + filtros opcionais
    G0 = read_gexf_simple(gexf_path)
    Gf = maybe_filter_edges_by_weight(G0, args.min_weight)
    Gf = maybe_limit_nodes(Gf, args.max_nodes)

    # 2) Grafo não-direcionado para layout & comunidades
    GU = to_undirected_simple(Gf)
    if args.keep_gcc:
        GU = keep_giant_component(GU)

    # 3) Comunidades e grau
    comm_map = detect_communities(GU, seed=args.seed)
    deg_map  = dict(GU.degree())

    # 4) Layout (100% NetworkX)
    pos = build_layout(GU, seed=args.seed)

    # 5) Salvar CSVs de métricas + arestas
    pd.DataFrame({
        "id": list(GU.nodes()),
        "degree": [deg_map[n] for n in GU.nodes()],
        "community": [comm_map[n] for n in GU.nodes()],
        "x": [pos[n][0] for n in GU.nodes()],
        "y": [pos[n][1] for n in GU.nodes()],
    }).to_csv(outdir / "node_metrics.csv", index=False)

    pd.DataFrame({
        "source": [u for u, v in Gf.edges()],
        "target": [v for u, v in Gf.edges()],
        "weight": [Gf[u][v].get("weight", None) for u, v in Gf.edges()],
    }).to_csv(outdir / "edge_metrics.csv", index=False)

    # 6) Desenho estático (PNG + SVG) — sem nomes e com legenda das arestas por sinal
    out_png = outdir / "viz_presentation.png"
    out_svg = outdir / "viz_presentation.svg"
    title = f"{stem} — comunidades (cores) e arestas por sinal do peso"
    draw_static(Gf, GU, pos, comm_map, deg_map, out_png, out_svg, title=title)

    # 7) Interativo (HTML), se disponível
    if args.interactive in ("both", "html"):
        out_html = outdir / "viz_presentation.html"
        draw_interactive(Gf, GU, pos, comm_map, deg_map, out_html, title=title)

    # 8) README com parâmetros
    (outdir / "README_viz.txt").write_text(
        f"""Grafo: {gexf_path}
Parâmetros:
  keep_gcc={args.keep_gcc}
  min_weight={args.min_weight}
  max_nodes={args.max_nodes}
  seed={args.seed}
  interactive={args.interactive}
Notas:
  - Sem rótulos de nós/arestas (por solicitação).
  - Cores das arestas: verde (w>0), vermelho (w<0), cinza-escuro (w=0).
  - Cores dos nós indicam comunidades; tamanhos indicam grau (não-direcionado).
  - Layout 100% NetworkX: spring leve (<=3000 nós) ou coarse-to-fine (>3000 nós).
""",
        encoding="utf-8"
    )

    print("Visualizações salvas em:", outdir.resolve())
    print(" -", out_png.name, "|", out_svg.name)
    if args.interactive in ("both", "html"):
        print(" - viz_presentation.html (se plotly estiver instalado)")
    print(" - node_metrics.csv, edge_metrics.csv, README_viz.txt")

if __name__ == "__main__":
    main()
