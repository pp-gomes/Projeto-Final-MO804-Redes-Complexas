# Projeto Final — MO804 Redes Complexas

**Equilíbrio estrutural em redes com sinais (Reddit r/politics)**

Este repositório contém o pipeline completo — da coleta ao relatório — para analisar **equilíbrio estrutural** em redes **assinadas** (pesos em `[-1, 1]`) a partir de interações do Reddit.

> Principais entregas:
>
> * Construção do grafo (`.gexf`) a partir de comentários do Reddit.
> * Métricas estruturais básicas (|V|, |E|, densidade, clusterização, distâncias, componentes).
> * Análise de equilíbrio por tríades com **índice BT** (assinado + magnitudes).
> * **Modelo nulo** por embaralhamento de pesos e comparativos observado × nulo.
> * Visualização para apresentação (estática e interativa).
> * Histograma dos **pesos das arestas** em `[-1,1]`.
> * Relatório final em LaTeX.

---

## Sumário

* [Estrutura do repositório](#estrutura-do-repositório)
* [Instalação](#instalação)
* [Credenciais do Reddit (PRAW)](#credenciais-do-reddit-praw)
* [Fluxo de trabalho (pipeline)](#fluxo-de-trabalho-pipeline)
* [Scripts e saídas](#scripts-e-saídas)
* [Interpretação de BT](#interpretação-de-bt)
* [Reprodutibilidade](#reprodutibilidade)
* [Solução de problemas (FAQ)](#solução-de-problemas-faq)
* [Licença e citação](#licença-e-citação)

---

## Estrutura do repositório

```
Projeto-Final-MO804-Redes-Complexas/
├── extract_comments.py            # coleta do Reddit e exporta .gexf
├── merge_graphs.py                # (opcional) mescla múltiplos .gexf
├── graph_basic_stats.py           # métricas básicas do grafo
├── bt_threshold_plus_signed.py    # BT assinado + histogramas + GEXF filtrado por |BT|
├── bt_null_model.py               # modelo nulo (embaralha pesos) + histogramas
├── bt_compare_plots.py            # comparativos observado × nulo
├── edge_weight_hist.py            # histograma de pesos de arestas em [-1,1]
├── graph_presentation_viz.py      # visualização (PNG/SVG e opcional HTML)
├── docs/
│   └── relatorio_final.tex        # relatório final (LaTeX)
└── data/                          # (sugestão) colocar aqui seus .gexf e saídas
```

As saídas de cada script são gravadas automaticamente em pastas do tipo:

* `<stem>.gexf__basics`, `<stem>.gexf__bt_signed`, `<stem>.gexf__null`, `<stem>.gexf__compare`, `<stem>.gexf__weights`, `<stem>.gexf__viz`
  onde `stem` é o nome do arquivo `.gexf` (ex.: `politics.gexf` → `politics.gexf__viz/`).

---

## Instalação

Recomenda-se **Python 3.10+** (3.11 funciona bem).

> Observação: nenhuma dependência nativa pesada é exigida (sem `pygraphviz`/`fa2`).

```bash
# 1) (opcional) criar ambiente virtual
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 2) instalar dependências
pip install --upgrade pip
pip install networkx pandas numpy matplotlib tqdm praw plotly python-louvain
# SciPy é opcional; se não instalado, o layout cai no modo leve do NetworkX
# pip install scipy
```

---

## Credenciais do Reddit (PRAW)

Para usar `extract_comments.py`, configure um app no [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) e forneça as credenciais via:

**Opção A — `praw.ini` na raiz do projeto**

```ini
[default]
client_id=SEU_CLIENT_ID
client_secret=SEU_CLIENT_SECRET
user_agent=mo804-balance/0.1 by u/SEU_USUARIO
```

**Opção B — Variáveis de ambiente**

```bash
set PRAW_CLIENT_ID=...
set PRAW_CLIENT_SECRET=...
set PRAW_USER_AGENT=mo804-balance/0.1 by u/SEU_USUARIO
```

---

## Fluxo de trabalho (pipeline)

1. **Coleta e GEXF**

```bash
python extract_comments.py --subreddits r/politics --post_limit 30 --min_comments 5
# (opcional) se tiver vários .gexf em uma pasta:
python merge_graphs.py --indir ./data/gexf_multi --out ./data/politics.gexf
```

2. **Métricas básicas**

```bash
python graph_basic_stats.py --gexf ./data/politics.gexf
```

3. **Equilíbrio (BT assinado) + GEXF filtrado**

```bash
python bt_threshold_plus_signed.py --gexf ./data/politics.gexf --abs-threshold 0.3
```

4. **Modelo nulo + comparativos**

```bash
python bt_null_model.py --gexf ./data/politics.gexf --n-iters 100 --shuffle-mode global
python bt_compare_plots.py --gexf ./data/politics.gexf
```

5. **Visualização para apresentação**

```bash
python graph_presentation_viz.py --gexf ./data/politics.gexf --keep-gcc --min-weight 0.1 --interactive both
```

6. **Histograma de pesos de arestas**

```bash
python edge_weight_hist.py --gexf ./data/politics.gexf --step 0.1
```

---

## Scripts e saídas

### `extract_comments.py`

* Varre `subreddit.new()` com PRAW, constrói arestas (autor → pai) e grava `.gexf`.
* Parâmetros úteis: `--subreddits`, `--post_limit`, `--min_comments`, `--max_comments`, `--max_posts`.
* Saída: `*.gexf` na pasta escolhida.

### `merge_graphs.py` (opcional)

* Mescla múltiplos `.gexf` em um único grafo.
* Saída: `merged.gexf`.

### `graph_basic_stats.py`

* Calcula: |V|, |E|, graus médios, densidade, clusterização global/local, distâncias e diâmetro (GCC), componentes.
* Saídas em `<stem>.gexf__basics/`:

  * `summary.json`, `clustering_by_node.csv`, `component_sizes.csv`
  * `degree_distribution.png`, `hist_clustering.png` (etc.)

### `bt_threshold_plus_signed.py`

* Calcula **BT** por tríade (produto dos três pesos).
* Classificação **assinada**:

  * **BT > 0.5**: fortemente balanceada
  * **0 < BT ≤ 0.5**: levemente balanceada
  * **−0.5 < BT ≤ 0**: levemente desbalanceada
  * **BT ≤ −0.5**: fortemente desbalanceada
* Gera histogramas de `BT` e `|BT|` e exporta GEXF **filtrado** por `|BT| ≥ θ`.
* Saídas em `<stem>.gexf__bt_signed/`:

  * `triads_gexf_signed.csv`, `triads_summary_signed.csv`
  * `hist_BT_gexf.png`, `hist_absBT_gexf.png`
  * `filtered_BT_geq_<θ>.gexf`

### `bt_null_model.py`

* Embaralha pesos (modos: `global`, `per_source`, `signs_only`) e recalcula BT/|BT|.
* Saídas em `<stem>.gexf__null/`: histogramas agregados, estatísticas do nulo.

### `bt_compare_plots.py`

* Lê observado × nulo e produz comparativos:
* Saídas em `<stem>.gexf__compare/`:

  * `gexf_compare_BT.png`, `gexf_compare_absBT.png`, tabelas resumo.

### `edge_weight_hist.py`

* Extrai `weight` por aresta (fallback: `tanh(score/10)`), plota histograma em `[-1,1]`.
* Saídas em `<stem>.gexf__weights/`:

  * `weights.csv`, `weight_hist.csv`, `weight_hist.png`, `summary.json`.

### `graph_presentation_viz.py`

* Visualização de apresentação (estática + interativa).
* **Nós**: cores por comunidade; tamanhos por grau.
* **Arestas**: **verde** (w>0), **vermelho** (w<0), **cinza-escuro** (w=0).
* Opções: `--keep-gcc`, `--min-weight`, `--max-nodes`, `--interactive {both,html,none}`.
* Saídas em `<stem>.gexf__viz/`:

  * `viz_presentation.png`, `viz_presentation.svg`, `viz_presentation.html` (se `plotly`),
  * `node_metrics.csv`, `edge_metrics.csv`, `README_viz.txt`.

---

## Interpretação de BT

* **Sinal** determina **equilíbrio vs desbalanceamento** (positivo tende a equilíbrio; negativo a antagonismo).
* **Magnitude** (|BT|) determina **estabilidade**: tríades com |BT| baixo são instáveis (mesmo se BT>0).
* Limiar empírico adotado neste projeto:

  * `BT > 0.5` (forte balanceada) / `0 < BT ≤ 0.5` (leve balanceada)
  * `−0.5 < BT ≤ 0` (leve desbalanceada) / `BT ≤ −0.5` (forte desbalanceada)

---

## Reprodutibilidade

```bash
# Estatísticas básicas
python graph_basic_stats.py --gexf ./data/politics.gexf

# BT + GEXF filtrado por |BT|≥0.3
python bt_threshold_plus_signed.py --gexf ./data/politics.gexf --abs-threshold 0.3

# Modelo nulo (100 iterações, embaralhamento global)
python bt_null_model.py --gexf ./data/politics.gexf --n-iters 100 --shuffle-mode global

# Comparativos observado × nulo
python bt_compare_plots.py --gexf ./data/politics.gexf

# Visual para apresentação (maior componente; |w|≥0.1)
python graph_presentation_viz.py --gexf ./data/politics.gexf --keep-gcc --min-weight 0.1 --interactive both

# Histograma de pesos de arestas
python edge_weight_hist.py --gexf ./data/politics.gexf --step 0.1
```

---

## Solução de problemas (FAQ)

**PRAW 401 (Unauthorized)**

* Verificar `client_id`, `client_secret`, `user_agent` (em `praw.ini` ou variáveis de ambiente).
* Checar se o app está como “script” e se o Reddit não solicitou rotinas anti-abuso.
* Reduzir `--post_limit`/`--max_posts` se encostar em rate limit.

**Layout muito lento / `KeyboardInterrupt`**

* Usar `graph_presentation_viz.py` com `--max-nodes` (ex.: 2000) e/ou `--min-weight` para filtrar arestas fracas.
* Manter `--keep-gcc` para focar somente na maior componente.

**Windows paths**

* Usar aspas em caminhos com espaço (`"--gexf ./data/politics.gexf"`).
* Evitar barras invertidas duplicadas; preferir `/` ou `Path` relativo.

**Dependências nativas (pygraphviz/fa2)**

* Não são necessárias neste projeto; o layout é 100% NetworkX.

---

## Licença e citação

* Licença: defina a licença em `LICENSE` (sugestão: MIT ou BSD-3-Clause).
* Se este trabalho for utilizado em publicações, citar como:

  > Gomes do Carmo, P.P. (2025). *Projeto Final — MO804 Redes Complexas: Equilíbrio estrutural em redes com sinais (Reddit r/politics).* Repositório GitHub: [https://github.com/pp-gomes/Projeto-Final-MO804-Redes-Complexas](https://github.com/pp-gomes/Projeto-Final-MO804-Redes-Complexas)

---

**Contato:** Pedro Paulo Gomes do Carmo
**Repositório:** [https://github.com/pp-gomes/Projeto-Final-MO804-Redes-Complexas](https://github.com/pp-gomes/Projeto-Final-MO804-Redes-Complexas)

---
