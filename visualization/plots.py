"""
visualization/plots.py
Geração de todos os gráficos e dashboard HTML interativo com Plotly.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from wordcloud import WordCloud, STOPWORDS

from config import OUTPUT_DIR, FIGURE_DPI

logger = logging.getLogger(__name__)
OUT = Path(OUTPUT_DIR)
OUT.mkdir(exist_ok=True)

# Paleta dark fantasy
PALETTE = {
    "very_positive": "#27ae60",
    "positive":      "#2ecc71",
    "neutral":       "#95a5a6",
    "negative":      "#e74c3c",
    "very_negative": "#c0392b",
}
GAME_COLORS = px.colors.qualitative.Dark24

matplotlib.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor":   "#16213e",
    "axes.edgecolor":   "#0f3460",
    "axes.labelcolor":  "#e0e0e0",
    "text.color":       "#e0e0e0",
    "xtick.color":      "#e0e0e0",
    "ytick.color":      "#e0e0e0",
    "grid.color":       "#0f3460",
    "font.family":      "DejaVu Sans",
})


# ─────────────────────────────────────────────────────────────────────────────
#  1. Barras – Ranking de sentimento por jogo
# ─────────────────────────────────────────────────────────────────────────────

def plot_sentiment_ranking(summary: pd.DataFrame) -> str:
    fig = px.bar(
        summary.sort_values("mean_compound"),
        x="mean_compound", y="game_name",
        orientation="h",
        color="mean_compound",
        color_continuous_scale=["#c0392b", "#e67e22", "#f1c40f", "#2ecc71"],
        text=summary.sort_values("mean_compound")["mean_compound"].apply(lambda v: f"{v:+.3f}"),
        title="🏆 Ranking de Sentimento Médio — Jogos Soulslike",
        labels={"mean_compound": "Score de Sentimento (VADER)", "game_name": ""},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        template="plotly_dark", height=600,
        coloraxis_showscale=False,
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
        font=dict(color="#e0e0e0"),
    )
    path = str(OUT / "01_sentiment_ranking.html")
    fig.write_html(path)
    logger.info(f"Salvo: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  2. Distribuição de labels por jogo (stacked bar)
# ─────────────────────────────────────────────────────────────────────────────

def plot_label_distribution(df: pd.DataFrame) -> str:
    order = ["very_positive", "positive", "neutral", "negative", "very_negative"]
    dist = (
        df.groupby(["game_name", "label"])
        .size()
        .reset_index(name="count")
    )
    totals = df.groupby("game_name").size().rename("total")
    dist = dist.merge(totals, on="game_name")
    dist["pct"] = dist["count"] / dist["total"] * 100
    dist["label"] = pd.Categorical(dist["label"], categories=order, ordered=True)
    dist = dist.sort_values("label")

    fig = px.bar(
        dist,
        x="pct", y="game_name",
        color="label",
        orientation="h",
        color_discrete_map=PALETTE,
        category_orders={"label": order},
        title="📊 Distribuição de Sentimento por Jogo (%)",
        labels={"pct": "% de Reviews", "game_name": ""},
        text=dist["pct"].apply(lambda v: f"{v:.0f}%"),
    )
    fig.update_traces(textposition="inside")
    fig.update_layout(
        barmode="stack", template="plotly_dark", height=650,
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
        legend_title="Sentimento",
    )
    path = str(OUT / "02_label_distribution.html")
    fig.write_html(path)
    logger.info(f"Salvo: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  3. Scatter — Reviews vs Sentimento (popularidade)
# ─────────────────────────────────────────────────────────────────────────────

def plot_popularity_vs_sentiment(summary: pd.DataFrame) -> str:
    fig = px.scatter(
        summary,
        x="total_reviews", y="mean_compound",
        size="total_reviews",
        color="mean_compound",
        color_continuous_scale=["#c0392b", "#f1c40f", "#27ae60"],
        hover_name="game_name",
        text="game_name",
        title="🔵 Popularidade vs Sentimento",
        labels={
            "total_reviews": "Nº de Reviews",
            "mean_compound": "Sentimento Médio",
        },
    )
    fig.update_traces(textposition="top center", marker=dict(opacity=0.8))
    fig.update_layout(
        template="plotly_dark", height=600,
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
        coloraxis_showscale=False,
    )
    path = str(OUT / "03_popularity_vs_sentiment.html")
    fig.write_html(path)
    logger.info(f"Salvo: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  4. Heatmap de Aspectos por Jogo
# ─────────────────────────────────────────────────────────────────────────────

def plot_aspect_heatmap(df: pd.DataFrame) -> str:
    aspect_cols = [c for c in df.columns if c.startswith("aspect_")]
    if not aspect_cols:
        return ""

    aspect_by_game = df.groupby("game_name")[aspect_cols].mean()
    aspect_by_game.columns = [c.replace("aspect_", "").title() for c in aspect_cols]

    fig = px.imshow(
        aspect_by_game,
        color_continuous_scale=["#c0392b", "#2c3e50", "#27ae60"],
        color_continuous_midpoint=0,
        title="🎯 Heatmap de Sentimento por Aspecto e Jogo",
        labels=dict(x="Aspecto", y="Jogo", color="Sentimento"),
        aspect="auto",
    )
    fig.update_layout(
        template="plotly_dark", height=600,
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
    )
    path = str(OUT / "04_aspect_heatmap.html")
    fig.write_html(path)
    logger.info(f"Salvo: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  5. Série temporal de sentimento
# ─────────────────────────────────────────────────────────────────────────────

def plot_temporal_sentiment(ts: pd.DataFrame, top_games: int = 6) -> str:
    if ts.empty:
        return ""

    top = (
        ts.groupby("game_name")["review_count"].sum()
        .nlargest(top_games).index.tolist()
    )
    ts_top = ts[ts["game_name"].isin(top)].copy()
    ts_top["year_month_dt"] = pd.to_datetime(ts_top["year_month"], format="%Y-%m", errors="coerce")

    fig = px.line(
        ts_top,
        x="year_month_dt", y="mean_sentiment",
        color="game_name",
        title=f"📈 Evolução do Sentimento ao Longo do Tempo (Top {top_games} jogos)",
        labels={"year_month_dt": "Mês", "mean_sentiment": "Sentimento Médio"},
        markers=True,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#95a5a6", annotation_text="Neutro")
    fig.update_layout(
        template="plotly_dark", height=550,
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
    )
    path = str(OUT / "05_temporal_sentiment.html")
    fig.write_html(path)
    logger.info(f"Salvo: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  6. Word Clouds por jogo (top 6 jogos)
# ─────────────────────────────────────────────────────────────────────────────

def plot_wordclouds(df: pd.DataFrame, top_n: int = 6) -> str:
    stop = STOPWORDS | {
        "game", "play", "played", "will", "really", "one", "like",
        "just", "good", "great", "get", "also", "feel", "time",
    }
    top_games = df["game_name"].value_counts().head(top_n).index.tolist()

    n_cols = 3
    n_rows = (len(top_games) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten()

    for i, game in enumerate(top_games):
        text = " ".join(df[df["game_name"] == game]["review"].dropna().astype(str))
        wc = WordCloud(
            width=800, height=400,
            background_color="#1a1a2e",
            colormap="YlOrRd",
            stopwords=stop,
            max_words=80,
            collocations=False,
        ).generate(text)
        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].set_title(game, fontsize=13, pad=8, color="#e0e0e0")
        axes[i].axis("off")

    for j in range(len(top_games), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("☁️  Word Clouds — O que os jogadores falam sobre cada jogo",
                 fontsize=16, color="#e0e0e0", y=1.01)
    plt.tight_layout()
    path = str(OUT / "06_wordclouds.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    logger.info(f"Salvo: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  7. Tópicos LDA — Visualização
# ─────────────────────────────────────────────────────────────────────────────

def plot_lda_topics(lda_results: dict) -> str:
    topics = lda_results["topics"]
    topic_sentiment = lda_results.get("topic_sentiment", pd.DataFrame())

    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=[t["label"] for t in topics],
    )

    for idx, topic in enumerate(topics):
        row = idx // 4 + 1
        col = idx % 4 + 1
        words = topic["words"][:8]
        weights = topic["weights"][:8]
        fig.add_trace(
            go.Bar(
                x=weights, y=words,
                orientation="h",
                marker_color=GAME_COLORS[idx % len(GAME_COLORS)],
                name=topic["label"],
                showlegend=False,
            ),
            row=row, col=col,
        )

    fig.update_layout(
        title="🗂️  Tópicos Descobertos por LDA nas Reviews Soulslike",
        template="plotly_dark", height=700,
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
    )
    path = str(OUT / "07_lda_topics.html")
    fig.write_html(path)
    logger.info(f"Salvo: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  8. Clusters K-Means
# ─────────────────────────────────────────────────────────────────────────────

def plot_kmeans_clusters(kmeans_results: dict) -> str:
    df = kmeans_results["df"]
    stats = kmeans_results["cluster_stats"]
    terms = kmeans_results["cluster_terms"]

    # Adiciona termos ao label
    stats = stats.copy()
    stats["top_terms"] = stats["cluster"].apply(
        lambda c: ", ".join(terms.get(c, [])[:5])
    )

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "bar"}, {"type": "scatter"}]],
        subplot_titles=["Tamanho e Sentimento por Cluster",
                        "Distribuição de Sentimento por Cluster"],
    )

    # Barras
    fig.add_trace(go.Bar(
        x=stats["label"], y=stats["n_reviews"],
        marker_color=[GAME_COLORS[i] for i in range(len(stats))],
        name="Nº Reviews",
        text=stats["mean_sentiment"].apply(lambda v: f"Sent: {v:+.3f}"),
        textposition="auto",
    ), row=1, col=1)

    # Violin por cluster
    for cid, label in df[["cluster", "cluster_label"]].drop_duplicates().values:
        sub = df[df["cluster"] == cid]["compound"]
        fig.add_trace(go.Violin(
            y=sub, name=str(label), box_visible=True,
            meanline_visible=True,
            line_color=GAME_COLORS[int(cid) % len(GAME_COLORS)],
        ), row=1, col=2)

    fig.update_layout(
        title=f"🔵 Clusters K-Means de Reviews (k={kmeans_results['best_k']}, "
              f"silhouette={kmeans_results['silhouette']:.3f})",
        template="plotly_dark", height=550,
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
        showlegend=False,
    )
    path = str(OUT / "08_kmeans_clusters.html")
    fig.write_html(path)
    logger.info(f"Salvo: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  9. Box Plot — Playtime vs Sentimento
# ─────────────────────────────────────────────────────────────────────────────

def plot_playtime_sentiment(df: pd.DataFrame) -> str:
    if "playtime_forever_h" not in df.columns:
        return ""

    df2 = df.copy()
    df2["playtime_bucket"] = pd.cut(
        df2["playtime_forever_h"].clip(0, 500),
        bins=[0, 10, 50, 100, 250, 500],
        labels=["0-10h", "10-50h", "50-100h", "100-250h", "250-500h"],
    )

    fig = px.box(
        df2.dropna(subset=["playtime_bucket"]),
        x="playtime_bucket", y="compound",
        color="playtime_bucket",
        title="⏱️  Tempo de Jogo vs Sentimento da Review",
        labels={"playtime_bucket": "Horas de Jogo", "compound": "Score de Sentimento"},
        points=False,
        color_discrete_sequence=GAME_COLORS,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#95a5a6")
    fig.update_layout(
        template="plotly_dark", height=500,
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
        showlegend=False,
    )
    path = str(OUT / "09_playtime_sentiment.html")
    fig.write_html(path)
    logger.info(f"Salvo: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  10. Dashboard HTML consolidado
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard_html(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    rankings: dict,
    ts: pd.DataFrame,
    lda_results: dict,
    kmeans_results: dict,
) -> str:
    """Gera um dashboard HTML completo com todas as visualizações embutidas."""
    logger.info("Construindo dashboard HTML…")

    # Gera todos os gráficos como divs inline
    def fig_html(fig) -> str:
        return fig.to_html(full_html=False, include_plotlyjs=False)

    # ── Gráfico 1: Ranking ──
    ranking_fig = px.bar(
        summary.sort_values("mean_compound"),
        x="mean_compound", y="game_name", orientation="h",
        color="mean_compound",
        color_continuous_scale=["#c0392b","#e67e22","#f1c40f","#2ecc71"],
        text=summary.sort_values("mean_compound")["mean_compound"].apply(lambda v: f"{v:+.3f}"),
        title="🏆 Ranking de Sentimento Médio",
    )
    ranking_fig.update_traces(textposition="outside")
    ranking_fig.update_layout(template="plotly_dark", height=500, coloraxis_showscale=False,
                              plot_bgcolor="#1a1a2e", paper_bgcolor="#0d0d1a")

    # ── Gráfico 2: Distribuição de labels ──
    order = ["very_positive","positive","neutral","negative","very_negative"]
    dist = df.groupby(["game_name","label"]).size().reset_index(name="count")
    totals = df.groupby("game_name").size().rename("total")
    dist = dist.merge(totals, on="game_name")
    dist["pct"] = dist["count"] / dist["total"] * 100
    dist["label"] = pd.Categorical(dist["label"], categories=order, ordered=True)
    dist = dist.sort_values("label")
    label_fig = px.bar(
        dist, x="pct", y="game_name", color="label", orientation="h",
        color_discrete_map=PALETTE, category_orders={"label": order},
        title="📊 Distribuição de Sentimento por Jogo",
        text=dist["pct"].apply(lambda v: f"{v:.0f}%"),
    )
    label_fig.update_traces(textposition="inside")
    label_fig.update_layout(barmode="stack", template="plotly_dark", height=600,
                            plot_bgcolor="#1a1a2e", paper_bgcolor="#0d0d1a")

    # ── Gráfico 3: Heatmap de aspectos ──
    aspect_cols = [c for c in df.columns if c.startswith("aspect_")]
    aspect_html = ""
    if aspect_cols:
        aspect_by_game = df.groupby("game_name")[aspect_cols].mean()
        aspect_by_game.columns = [c.replace("aspect_","").title() for c in aspect_cols]
        aspect_fig = px.imshow(
            aspect_by_game,
            color_continuous_scale=["#c0392b","#2c3e50","#27ae60"],
            color_continuous_midpoint=0,
            title="🎯 Heatmap de Sentimento por Aspecto",
            aspect="auto",
        )
        aspect_fig.update_layout(template="plotly_dark", height=550,
                                 plot_bgcolor="#1a1a2e", paper_bgcolor="#0d0d1a")
        aspect_html = fig_html(aspect_fig)

    # ── Gráfico 4: Temporal ──
    temporal_html = ""
    if not ts.empty:
        top6 = ts.groupby("game_name")["review_count"].sum().nlargest(6).index.tolist()
        ts6 = ts[ts["game_name"].isin(top6)].copy()
        ts6["dt"] = pd.to_datetime(ts6["year_month"], format="%Y-%m", errors="coerce")
        temp_fig = px.line(
            ts6, x="dt", y="mean_sentiment", color="game_name",
            title="📈 Evolução do Sentimento ao Longo do Tempo",
            markers=True,
        )
        temp_fig.add_hline(y=0, line_dash="dash", line_color="#95a5a6")
        temp_fig.update_layout(template="plotly_dark", height=500,
                               plot_bgcolor="#1a1a2e", paper_bgcolor="#0d0d1a")
        temporal_html = fig_html(temp_fig)

    # ── Gráfico 5: LDA topics ──
    topics = lda_results["topics"]
    topic_fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=[t["label"] for t in topics[:8]],
    )
    for idx, topic in enumerate(topics[:8]):
        row, col = idx // 4 + 1, idx % 4 + 1
        topic_fig.add_trace(go.Bar(
            x=topic["weights"][:6], y=topic["words"][:6], orientation="h",
            marker_color=GAME_COLORS[idx % len(GAME_COLORS)], showlegend=False,
        ), row=row, col=col)
    topic_fig.update_layout(
        title="🗂️  Tópicos LDA — O que os jogadores discutem",
        template="plotly_dark", height=650,
        plot_bgcolor="#1a1a2e", paper_bgcolor="#0d0d1a",
    )

    # ── Gráfico 6: K-Means ──
    km_stats = kmeans_results["cluster_stats"]
    km_fig = px.bar(
        km_stats, x="label", y="n_reviews",
        color="mean_sentiment",
        color_continuous_scale=["#c0392b","#f1c40f","#27ae60"],
        title=f"🔵 Clusters de Reviews (k={kmeans_results['best_k']})",
        text=km_stats["mean_sentiment"].apply(lambda v: f"{v:+.3f}"),
    )
    km_fig.update_traces(textposition="outside")
    km_fig.update_layout(template="plotly_dark", height=450,
                         plot_bgcolor="#1a1a2e", paper_bgcolor="#0d0d1a",
                         coloraxis_showscale=False)

    # ── Tabela: Top reviews positivas ──
    top_pos = df.nlargest(5, "compound")[["game_name","review","compound"]].copy()
    top_pos["review"] = top_pos["review"].str[:200] + "…"
    top_pos["compound"] = top_pos["compound"].round(3)
    table_fig = go.Figure(go.Table(
        header=dict(
            values=["🎮 Jogo","💬 Review (trecho)","⭐ Score"],
            fill_color="#0f3460", font=dict(color="white", size=12),
        ),
        cells=dict(
            values=[top_pos[c] for c in ["game_name","review","compound"]],
            fill_color=[["#16213e","#1a1a2e"]*10],
            font=dict(color="#e0e0e0", size=11),
            align=["left","left","center"],
        ),
    ))
    table_fig.update_layout(title="🌟 Reviews Mais Positivas", height=380,
                            paper_bgcolor="#0d0d1a")

    # ── Métricas resumo ──
    total_reviews = len(df)
    n_games = df["game_name"].nunique()
    best_game = summary.loc[summary["mean_compound"].idxmax(), "game_name"]
    most_reviewed = summary.loc[summary["total_reviews"].idxmax(), "game_name"]
    pct_positive = (df["label"].isin(["positive","very_positive"])).mean() * 100

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>⚔️  Soulslike Sentiment Analyzer — Dashboard</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@700&family=Rajdhani:wght@400;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #0d0d1a; --card: #16213e; --accent: #c0392b;
    --gold: #f39c12; --text: #e0e0e0; --border: #0f3460;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Rajdhani', sans-serif; }}
  header {{
    background: linear-gradient(135deg, #0d0d1a 0%, #1a0a0a 50%, #0d0d1a 100%);
    border-bottom: 2px solid var(--accent);
    padding: 2rem; text-align: center;
  }}
  header h1 {{
    font-family: 'Cinzel', serif; font-size: 2.4rem;
    color: var(--gold); letter-spacing: 3px;
    text-shadow: 0 0 20px rgba(192,57,43,0.6);
  }}
  header p {{ color: #95a5a6; margin-top: 0.5rem; font-size: 1rem; }}
  .metrics {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem; padding: 1.5rem 2rem;
  }}
  .metric {{
    background: var(--card); border: 1px solid var(--border);
    border-top: 3px solid var(--accent); border-radius: 8px;
    padding: 1.2rem; text-align: center;
  }}
  .metric .value {{
    font-size: 2rem; font-weight: 700; color: var(--gold);
    font-family: 'Cinzel', serif;
  }}
  .metric .label {{
    font-size: 0.85rem; color: #95a5a6; margin-top: 0.3rem; text-transform: uppercase;
  }}
  .grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
    gap: 1.5rem; padding: 0 2rem 1.5rem;
  }}
  .card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
  }}
  .card-full {{ grid-column: 1 / -1; }}
  footer {{
    text-align: center; padding: 2rem; color: #555;
    border-top: 1px solid var(--border); margin-top: 2rem;
    font-size: 0.85rem;
  }}
</style>
</head>
<body>
<header>
  <h1>⚔️  SOULSLIKE SENTIMENT ANALYZER</h1>
  <p>Análise de Sentimento & Machine Learning — Reviews da Steam Community</p>
</header>

<div class="metrics">
  <div class="metric">
    <div class="value">{total_reviews:,}</div>
    <div class="label">Total de Reviews</div>
  </div>
  <div class="metric">
    <div class="value">{n_games}</div>
    <div class="label">Jogos Analisados</div>
  </div>
  <div class="metric">
    <div class="value">{pct_positive:.1f}%</div>
    <div class="label">Reviews Positivas</div>
  </div>
  <div class="metric">
    <div class="value" style="font-size:1.1rem">{best_game}</div>
    <div class="label">Melhor Sentimento</div>
  </div>
  <div class="metric">
    <div class="value" style="font-size:1.1rem">{most_reviewed}</div>
    <div class="label">Mais Comentado</div>
  </div>
  <div class="metric">
    <div class="value">{kmeans_results['best_k']}</div>
    <div class="label">Clusters K-Means</div>
  </div>
</div>

<div class="grid">
  <div class="card card-full">{fig_html(ranking_fig)}</div>
  <div class="card">{fig_html(label_fig)}</div>
  <div class="card">{aspect_html if aspect_html else "<p style='padding:2rem;color:#666'>Sem dados de aspecto</p>"}</div>
  <div class="card card-full">{temporal_html if temporal_html else ""}</div>
  <div class="card card-full">{fig_html(topic_fig)}</div>
  <div class="card">{fig_html(km_fig)}</div>
  <div class="card">{fig_html(table_fig)}</div>
</div>

<footer>
  ⚔️  Soulslike Sentiment Analyzer — dados coletados via Steam Reviews API &nbsp;|&nbsp;
  VADER + LDA + K-Means &nbsp;|&nbsp; Feito com Python & Plotly
</footer>
</body>
</html>"""

    path = str(OUT / "dashboard.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Dashboard salvo: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  Utilitário: exporta todos os gráficos
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_plots(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    rankings: dict,
    ts: pd.DataFrame,
    lda_results: dict,
    kmeans_results: dict,
) -> list[str]:
    paths = []
    paths.append(plot_sentiment_ranking(summary))
    paths.append(plot_label_distribution(df))
    paths.append(plot_popularity_vs_sentiment(summary))
    paths.append(plot_aspect_heatmap(df))
    paths.append(plot_temporal_sentiment(ts))
    paths.append(plot_wordclouds(df))
    paths.append(plot_lda_topics(lda_results))
    paths.append(plot_kmeans_clusters(kmeans_results))
    paths.append(plot_playtime_sentiment(df))
    paths.append(build_dashboard_html(df, summary, rankings, ts, lda_results, kmeans_results))
    return [p for p in paths if p]
