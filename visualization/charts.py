"""
Visualization Module
Generates all charts as interactive Plotly HTML files + static PNGs.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express       as px
from plotly.subplots import make_subplots
from wordcloud       import WordCloud
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Colour palette ──────────────────────────────────────────────────────────
PALETTE = {
    "Very Positive": "#2ecc71",
    "Positive":      "#82e0aa",
    "Neutral":       "#aab7b8",
    "Negative":      "#e59866",
    "Very Negative": "#e74c3c",
}
DARK_BG   = "#1a1a2e"
PANEL_BG  = "#16213e"
TEXT_COL  = "#eaeaea"
ACCENT    = "#e94560"

LAYOUT_BASE = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor=PANEL_BG,
    font=dict(color=TEXT_COL, family="Segoe UI, Arial"),
    margin=dict(l=60, r=40, t=70, b=60),
)


def _save(fig: go.Figure, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.write_html(path)
    print(f"  [chart] {path}")
    return path


# ─── 1  Ranking: mean sentiment per game ────────────────────────────────────
def chart_sentiment_ranking(game_stats: pd.DataFrame, out_dir: str) -> str:
    gs = game_stats.sort_values("mean_compound")
    colors = [PALETTE["Very Positive"] if v >= 0.5
              else PALETTE["Positive"] if v >= 0.1
              else PALETTE["Neutral"] if v >= -0.1
              else PALETTE["Negative"] for v in gs["mean_compound"]]

    fig = go.Figure(go.Bar(
        x=gs["mean_compound"], y=gs["game"],
        orientation="h",
        marker_color=colors,
        text=gs["mean_compound"].round(3),
        textposition="outside",
        customdata=gs[["n_reviews","pct_positive"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Sentiment: %{x:.3f}<br>"
            "Reviews: %{customdata[0]}<br>"
            "% Positive: %{customdata[1]:.1%}<extra></extra>"
        ),
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="🏆 Soulslike Games — Sentiment Ranking (VADER compound)",
                   font=dict(size=18)),
        xaxis=dict(title="Mean Compound Score", gridcolor="#2a2a4a", zeroline=True,
                   zerolinecolor="#555", range=[-1, 1]),
        yaxis=dict(title="", tickfont=dict(size=11)),
        height=max(500, len(gs) * 40 + 100),
    )
    return _save(fig, out_dir, "01_sentiment_ranking.html")


# ─── 2  Stacked sentiment distribution per game ──────────────────────────────
def chart_sentiment_distribution(df: pd.DataFrame, out_dir: str) -> str:
    steam = df[df["source"] == "steam"]
    order = ["Very Negative","Negative","Neutral","Positive","Very Positive"]
    counts = (
        steam.groupby(["game","sentiment_label"]).size()
        .unstack(fill_value=0).reindex(columns=order, fill_value=0)
    )
    # sort by positivity
    counts["_pos"] = counts.get("Very Positive",0) + counts.get("Positive",0)
    counts = counts.sort_values("_pos", ascending=True).drop("_pos", axis=1)

    fig = go.Figure()
    for label in order:
        if label in counts.columns:
            fig.add_trace(go.Bar(
                name=label,
                y=counts.index.tolist(),
                x=counts[label],
                orientation="h",
                marker_color=PALETTE[label],
                hovertemplate=f"{label}: %{{x}}<extra></extra>",
            ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="📊 Sentiment Distribution by Game", font=dict(size=18)),
        barmode="stack",
        xaxis=dict(title="Number of Reviews", gridcolor="#2a2a4a"),
        yaxis=dict(title="", tickfont=dict(size=11)),
        legend=dict(bgcolor="#1a1a2e", bordercolor="#555"),
        height=max(500, len(counts) * 40 + 120),
    )
    return _save(fig, out_dir, "02_sentiment_distribution.html")


# ─── 3  Review volume ranking ────────────────────────────────────────────────
def chart_volume_ranking(game_stats: pd.DataFrame, out_dir: str) -> str:
    gs = game_stats.sort_values("n_reviews", ascending=True)
    fig = go.Figure(go.Bar(
        x=gs["n_reviews"], y=gs["game"],
        orientation="h",
        marker=dict(color=gs["mean_compound"], colorscale="RdYlGn",
                    cmin=-1, cmax=1, showscale=True,
                    colorbar=dict(title="Sentiment", tickfont=dict(color=TEXT_COL))),
        text=gs["n_reviews"], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Reviews: %{x}<br>Sentiment: %{marker.color:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="📈 Most Discussed Games (review volume × sentiment colour)",
                   font=dict(size=18)),
        xaxis=dict(title="Number of Reviews", gridcolor="#2a2a4a"),
        yaxis=dict(title=""),
        height=max(500, len(gs) * 40 + 100),
    )
    return _save(fig, out_dir, "03_volume_ranking.html")


# ─── 4  Sentiment over time ──────────────────────────────────────────────────
def chart_temporal_trend(trend_df: pd.DataFrame, out_dir: str) -> str:
    fig = go.Figure()
    colors = px.colors.qualitative.Safe
    for i, col in enumerate(trend_df.columns):
        fig.add_trace(go.Scatter(
            x=trend_df.index, y=trend_df[col],
            mode="lines+markers", name=col,
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate=f"<b>{col}</b><br>Month: %{{x}}<br>Sentiment: %{{y:.3f}}<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="#888", annotation_text="Neutral",
                  annotation_font_color="#888")
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="📅 Sentiment Over Time (Top Games)", font=dict(size=18)),
        xaxis=dict(title="Month", gridcolor="#2a2a4a", tickangle=-45),
        yaxis=dict(title="Mean Compound Score", gridcolor="#2a2a4a", range=[-1, 1]),
        legend=dict(bgcolor="#1a1a2e", bordercolor="#555"),
        height=500,
    )
    return _save(fig, out_dir, "04_temporal_trend.html")


# ─── 5  Topic distribution ───────────────────────────────────────────────────
def chart_topics(topics_df: pd.DataFrame, doc_topics: pd.Series,
                 df: pd.DataFrame, out_dir: str) -> str:
    topic_counts = doc_topics.value_counts().reset_index()
    topic_counts.columns = ["topic_id", "count"]
    topic_counts = topic_counts.merge(topics_df[["topic_id","label"]], on="topic_id")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Topic Prevalence", "Mean Sentiment per Topic"),
        specs=[[{"type":"pie"},{"type":"bar"}]],
    )
    fig.add_trace(go.Pie(
        labels=topic_counts["label"], values=topic_counts["count"],
        hole=0.4,
        marker=dict(colors=px.colors.qualitative.Pastel),
        textinfo="percent+label",
    ), row=1, col=1)

    # sentiment per topic
    merged = df.copy()
    merged["topic_id"] = doc_topics.values
    topic_sent = merged.groupby("topic_id")["compound"].mean().reset_index()
    topic_sent = topic_sent.merge(topics_df[["topic_id","label"]], on="topic_id")
    topic_sent = topic_sent.sort_values("compound")
    bar_colors = [PALETTE["Very Positive"] if v >= 0.5
                  else PALETTE["Positive"] if v >= 0.1
                  else PALETTE["Neutral"] if v >= -0.1
                  else PALETTE["Negative"] for v in topic_sent["compound"]]
    fig.add_trace(go.Bar(
        x=topic_sent["compound"], y=topic_sent["label"],
        orientation="h", marker_color=bar_colors,
        text=topic_sent["compound"].round(3), textposition="outside",
    ), row=1, col=2)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="🧠 NMF Topic Analysis", font=dict(size=18)),
        height=500,
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="#2a2a4a")
    fig.update_yaxes(tickfont=dict(size=10))
    return _save(fig, out_dir, "05_topics.html")


# ─── 6  Radar chart (top 8 games) ────────────────────────────────────────────
def chart_radar(radar_df: pd.DataFrame, out_dir: str) -> str:
    categories = ["Sentiment","Positivity","Review Volume","Avg Playtime","Consistency"]
    top = radar_df.nlargest(8, "Sentiment")
    colors = px.colors.qualitative.Safe

    fig = go.Figure()
    for i, (_, row) in enumerate(top.iterrows()):
        vals = [row[c] for c in categories] + [row[categories[0]]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=categories + [categories[0]],
            fill="toself", name=row["game"],
            line=dict(color=colors[i % len(colors)]),
            opacity=0.7,
        ))
    fig.update_layout(
        **LAYOUT_BASE,
        polar=dict(
            bgcolor=PANEL_BG,
            radialaxis=dict(visible=True, range=[0,1],
                            gridcolor="#333", tickfont=dict(color="#888")),
            angularaxis=dict(gridcolor="#333", tickfont=dict(color=TEXT_COL)),
        ),
        title=dict(text="🕷️ Multi-Metric Radar (Top 8 Games)", font=dict(size=18)),
        showlegend=True,
        legend=dict(bgcolor="#1a1a2e", bordercolor="#555"),
        height=550,
    )
    return _save(fig, out_dir, "06_radar.html")


# ─── 7  Cluster scatter ──────────────────────────────────────────────────────
def chart_clusters(df: pd.DataFrame, out_dir: str) -> str:
    if "cluster" not in df.columns:
        return ""
    steam = df[df["source"] == "steam"].copy()
    steam["cluster_str"] = "Cluster " + steam["cluster"].astype(str)

    fig = px.scatter(
        steam, x="compound", y="playtime_h",
        color="cluster_str",
        hover_data=["game","text"],
        title="🔍 Review Clusters (Sentiment × Playtime)",
        color_discrete_sequence=px.colors.qualitative.Safe,
        labels={"compound": "Sentiment Score", "playtime_h": "Playtime (h)"},
        opacity=0.7,
    )
    fig.update_layout(**LAYOUT_BASE, height=500)
    fig.update_traces(marker_size=6)
    return _save(fig, out_dir, "07_clusters.html")


# ─── 8  Compound score histogram ─────────────────────────────────────────────
def chart_sentiment_histogram(df: pd.DataFrame, out_dir: str) -> str:
    steam = df[df["source"] == "steam"]
    fig = px.histogram(
        steam, x="compound", nbins=50,
        color="sentiment_label",
        color_discrete_map=PALETTE,
        title="📉 Compound Score Distribution (all reviews)",
        labels={"compound": "VADER Compound Score", "count": "# Reviews"},
        category_orders={"sentiment_label":
            ["Very Negative","Negative","Neutral","Positive","Very Positive"]},
        barmode="overlay",
        opacity=0.7,
    )
    fig.add_vline(x=0, line_dash="dot", line_color="#eee",
                  annotation_text="Neutral", annotation_font_color="#eee")
    fig.update_layout(**LAYOUT_BASE, height=450, bargap=0.05)
    return _save(fig, out_dir, "08_histogram.html")


# ─── 9  Playtime vs sentiment scatter ────────────────────────────────────────
def chart_playtime_sentiment(df: pd.DataFrame, out_dir: str) -> str:
    steam = df[(df["source"]=="steam") & (df["playtime_h"] < 2000)].copy()
    fig = px.scatter(
        steam, x="playtime_h", y="compound",
        color="game", opacity=0.5,
        title="🎮 Playtime vs Sentiment (does more time = more love?)",
        labels={"playtime_h":"Playtime (hours)","compound":"Sentiment Score"},
        color_discrete_sequence=px.colors.qualitative.Safe,
        trendline="lowess",
    )
    fig.update_layout(**LAYOUT_BASE, height=500)
    fig.update_traces(marker_size=5)
    return _save(fig, out_dir, "09_playtime_vs_sentiment.html")


# ─── 10  Word clouds per game ────────────────────────────────────────────────
def chart_wordclouds(keywords_per_game: dict, out_dir: str) -> str:
    """Generate a grid of word clouds for all games."""
    n = len(keywords_per_game)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 5))
    fig.patch.set_facecolor("#1a1a2e")
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for i, (game, kws) in enumerate(keywords_per_game.items()):
        ax = axes[i]
        if not kws:
            ax.axis("off")
            continue
        freq = {w: v for w, v in kws}
        wc = WordCloud(
            width=500, height=300,
            background_color="#16213e",
            colormap="plasma",
            max_words=40,
            relative_scaling=0.6,
        ).generate_from_frequencies(freq)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(game, color="#eaeaea", fontsize=10, pad=8)
        ax.axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("☁️ Keyword Word Clouds by Game (TF-IDF)",
                 color="#eaeaea", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "10_wordclouds.png")
    plt.savefig(path, dpi=110, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"  [chart] {path}")
    return path


# ─── 11  Reddit vs Steam sentiment comparison ─────────────────────────────────
def chart_source_comparison(df: pd.DataFrame, out_dir: str) -> str:
    comp = (
        df.groupby(["source","sentiment_label"]).size()
        .reset_index(name="count")
    )
    order = ["Very Negative","Negative","Neutral","Positive","Very Positive"]
    fig = px.bar(
        comp, x="sentiment_label", y="count", color="source",
        barmode="group",
        title="🌐 Sentiment: Steam Reviews vs Reddit Posts",
        labels={"sentiment_label":"Sentiment","count":"Count","source":"Source"},
        category_orders={"sentiment_label": order},
        color_discrete_map={"steam":"#1b2838","reddit":"#ff4500"},
    )
    fig.update_layout(**LAYOUT_BASE, height=420)
    return _save(fig, out_dir, "11_steam_vs_reddit.html")


# ─── Generate all charts ─────────────────────────────────────────────────────
def generate_all(df, game_stats, trend_df, topics_df, doc_topics,
                 keywords_per_game, radar_df, out_dir):
    print("\n[Visualization] Generating charts...")
    paths = {}
    paths["ranking"]      = chart_sentiment_ranking(game_stats, out_dir)
    paths["distribution"] = chart_sentiment_distribution(df, out_dir)
    paths["volume"]       = chart_volume_ranking(game_stats, out_dir)
    if not trend_df.empty:
        paths["temporal"] = chart_temporal_trend(trend_df, out_dir)
    if topics_df is not None:
        paths["topics"]   = chart_topics(topics_df, doc_topics, df, out_dir)
    paths["radar"]        = chart_radar(radar_df, out_dir)
    if "cluster" in df.columns:
        paths["clusters"] = chart_clusters(df, out_dir)
    paths["histogram"]    = chart_sentiment_histogram(df, out_dir)
    paths["playtime"]     = chart_playtime_sentiment(df, out_dir)
    if keywords_per_game:
        paths["wordcloud"]= chart_wordclouds(keywords_per_game, out_dir)
    if df["source"].nunique() > 1:
        paths["sources"]  = chart_source_comparison(df, out_dir)
    return paths
