#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║          ⚔️  SOULSLIKE SENTIMENT ANALYZER  ⚔️               ║
║  Steam Reviews API + Reddit JSON API                         ║
║  VADER NLP · NMF Topics · KMeans Clustering · Plotly Charts  ║
╚══════════════════════════════════════════════════════════════╝

Usage
-----
  python main.py                  # full run (Steam + Reddit)
  python main.py --steam-only     # skip Reddit
  python main.py --max 100        # 100 reviews per game
  python main.py --load           # reload saved data, skip scraping
  python main.py --games 3        # only first 3 games (quick test)
"""

import argparse
import os
import sys
import time
import json

import pandas as pd
from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table
from rich         import print as rprint

# ─── make local modules importable ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    SOULSLIKE_GAMES, SOULSLIKE_SUBREDDITS, REDDIT_SEARCH_TERMS,
    STEAM_REVIEWS_PER_GAME, REDDIT_POSTS_PER_SEARCH,
    STEAM_REVIEW_LANGUAGE, REQUEST_DELAY_SECONDS,
    N_TOPICS, N_CLUSTERS, TOP_KEYWORDS, MIN_WORD_LENGTH,
    MAX_FEATURES_TFIDF, CUSTOM_STOP_WORDS, OUTPUT_DIR, DATA_DIR,
)
from scrapers.steam_scraper   import scrape_all_games
from scrapers.reddit_scraper  import scrape_reddit
from analysis.sentiment       import analyse_sentiment, compute_game_stats
from analysis.ml_analysis     import (
    extract_keywords_per_game, topic_modelling,
    cluster_reviews, cluster_summary, temporal_trend, radar_metrics,
)
from visualization.charts  import generate_all
from visualization.report  import generate_report

console = Console()

BANNER = """
[bold red]⚔[/bold red][bold white]  SOULSLIKE SENTIMENT ANALYZER[/bold white]  [bold red]⚔[/bold red]
[dim]Steam Reviews · Reddit · NLP · ML · Data Science[/dim]
"""


# ─── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Soulslike Sentiment Analyzer")
    p.add_argument("--steam-only",  action="store_true", help="Skip Reddit scraping")
    p.add_argument("--reddit-only", action="store_true", help="Skip Steam scraping")
    p.add_argument("--load",        action="store_true",
                   help="Load previously saved data instead of scraping")
    p.add_argument("--max",  type=int, default=STEAM_REVIEWS_PER_GAME,
                   help=f"Max reviews per game (default: {STEAM_REVIEWS_PER_GAME})")
    p.add_argument("--games",type=int, default=None,
                   help="Limit to first N games (useful for quick testing)")
    p.add_argument("--no-reddit-search", action="store_true",
                   help="Only collect top posts, skip search queries")
    p.add_argument("--out", default=OUTPUT_DIR, help="Output directory")
    return p.parse_args()


# ─── Data helpers ─────────────────────────────────────────────────────────────
def save_data(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False)
    console.print(f"[dim]Data saved → {path}[/dim]")


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        console.print(f"[red]No saved data found at {path}[/red]")
        sys.exit(1)
    df = pd.read_parquet(path)
    console.print(f"[green]Loaded {len(df):,} records from {path}[/green]")
    return df


# ─── Console reporting ────────────────────────────────────────────────────────
def print_sentiment_table(game_stats: pd.DataFrame):
    table = Table(title="🏆 Sentiment Ranking", show_header=True,
                  header_style="bold magenta", border_style="dim")
    table.add_column("#",         style="dim", width=4)
    table.add_column("Game",      min_width=28)
    table.add_column("Sentiment", justify="right")
    table.add_column("Label",     min_width=14)
    table.add_column("Reviews",   justify="right")
    table.add_column("% Pos",     justify="right")
    table.add_column("Playtime",  justify="right")

    for _, r in game_stats.iterrows():
        c = float(r.get("mean_compound", 0))
        if   c >=  0.5: col, lbl = "green",  "Very Positive"
        elif c >=  0.1: col, lbl = "cyan",   "Positive"
        elif c >= -0.1: col, lbl = "yellow", "Neutral"
        elif c >= -0.5: col, lbl = "orange1","Negative"
        else:           col, lbl = "red",    "Very Negative"
        table.add_row(
            str(int(r.get("rank", 0))),
            r["game"],
            f"[{col}]{c:+.3f}[/{col}]",
            f"[{col}]{lbl}[/{col}]",
            str(int(r.get("n_reviews", 0))),
            f"{r.get('pct_positive', 0):.1%}",
            f"{r.get('mean_playtime_h', 0) or 0:.1f}h",
        )
    console.print(table)


def print_topics(topics_df: pd.DataFrame):
    table = Table(title="🧠 NMF Topics", show_header=True,
                  header_style="bold magenta", border_style="dim")
    table.add_column("ID",    width=4)
    table.add_column("Label", min_width=30)
    table.add_column("Top Keywords", min_width=60)
    table.add_column("Weight", justify="right")
    for _, r in topics_df.iterrows():
        table.add_row(str(r["topic_id"]), r["label"],
                      f"[dim]{r['top_words'][:70]}[/dim]",
                      f"{r['mean_weight']:.4f}")
    console.print(table)


def print_cluster_summary(cs: pd.DataFrame):
    table = Table(title="🔍 Cluster Summary", show_header=True,
                  header_style="bold magenta", border_style="dim")
    for col in cs.columns:
        table.add_column(str(col).replace("_"," ").title())
    for _, r in cs.iterrows():
        table.add_row(*[str(v) for v in r.values])
    console.print(table)


# ─── Main pipeline ────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(DATA_DIR,  exist_ok=True)

    console.print(Panel(BANNER, expand=False, border_style="red"))

    data_path = os.path.join(DATA_DIR, "reviews.parquet")

    # ── 1. DATA COLLECTION ───────────────────────────────────────────────────
    if args.load:
        df_raw = load_data(data_path)
    else:
        records = []

        # Steam
        if not args.reddit_only:
            games = dict(list(SOULSLIKE_GAMES.items())[:args.games]
                         if args.games else SOULSLIKE_GAMES.items())
            console.rule("[bold cyan]Steam Reviews[/bold cyan]")
            steam_records = scrape_all_games(
                games,
                max_per_game=args.max,
                language=STEAM_REVIEW_LANGUAGE,
                delay=REQUEST_DELAY_SECONDS,
            )
            records.extend(steam_records)

        # Reddit
        if not args.steam_only:
            console.rule("[bold magenta]Reddit[/bold magenta]")
            search_terms = [] if args.no_reddit_search else REDDIT_SEARCH_TERMS
            reddit_records = scrape_reddit(
                SOULSLIKE_SUBREDDITS,
                search_terms,
                posts_per_search=REDDIT_POSTS_PER_SEARCH,
                delay=REQUEST_DELAY_SECONDS,
            )
            records.extend(reddit_records)

        if not records:
            console.print("[red]No data collected. Exiting.[/red]")
            sys.exit(1)

        df_raw = pd.DataFrame(records)
        save_data(df_raw, data_path)

    console.print(f"\n[bold]Total records:[/bold] {len(df_raw):,} "
                  f"([cyan]steam: {(df_raw['source']=='steam').sum()}[/cyan] / "
                  f"[magenta]reddit: {(df_raw['source']=='reddit').sum()}[/magenta])")

    # ── 2. SENTIMENT ANALYSIS ────────────────────────────────────────────────
    console.rule("[bold green]Sentiment Analysis[/bold green]")
    console.print("Running VADER with gaming domain lexicon...")
    df = analyse_sentiment(df_raw, text_col="text")

    # ── 3. GAME STATS ────────────────────────────────────────────────────────
    game_stats = compute_game_stats(df)
    print_sentiment_table(game_stats)

    # ── 4. TF-IDF KEYWORDS ───────────────────────────────────────────────────
    console.rule("[bold yellow]TF-IDF Keyword Extraction[/bold yellow]")
    console.print("Extracting top keywords per game...")
    keywords_per_game = extract_keywords_per_game(
        df, CUSTOM_STOP_WORDS, top_n=TOP_KEYWORDS, min_word_len=MIN_WORD_LENGTH
    )
    for game, kws in list(keywords_per_game.items())[:3]:
        top5 = ", ".join(w for w, _ in kws[:5])
        console.print(f"  [cyan]{game}[/cyan]: {top5}")

    # ── 5. TOPIC MODELLING ───────────────────────────────────────────────────
    console.rule("[bold yellow]NMF Topic Modelling[/bold yellow]")
    topics_df, doc_topics, _, _ = topic_modelling(
        df, CUSTOM_STOP_WORDS, n_topics=N_TOPICS, max_features=MAX_FEATURES_TFIDF
    )
    df["topic_id"] = doc_topics.values
    print_topics(topics_df)

    # ── 6. CLUSTERING ────────────────────────────────────────────────────────
    console.rule("[bold yellow]KMeans Clustering[/bold yellow]")
    console.print(f"Clustering into {N_CLUSTERS} groups (TF-IDF + LSA)...")
    df = cluster_reviews(df, CUSTOM_STOP_WORDS, n_clusters=N_CLUSTERS)
    cs = cluster_summary(df)
    print_cluster_summary(cs)

    # ── 7. TEMPORAL TREND ────────────────────────────────────────────────────
    console.rule("[bold yellow]Temporal Analysis[/bold yellow]")
    trend_df = temporal_trend(df)
    console.print(f"  Trend data: {len(trend_df)} months × {len(trend_df.columns)} games")

    # ── 8. RADAR METRICS ─────────────────────────────────────────────────────
    radar_df = radar_metrics(df, game_stats)

    # ── 9. VISUALIZATIONS ────────────────────────────────────────────────────
    console.rule("[bold green]Visualization[/bold green]")
    chart_paths = generate_all(
        df, game_stats, trend_df, topics_df, doc_topics,
        keywords_per_game, radar_df, args.out
    )

    # ── 10. FINAL REPORT ─────────────────────────────────────────────────────
    console.rule("[bold green]HTML Report[/bold green]")
    report_path = generate_report(df, game_stats, topics_df, chart_paths, args.out)

    # ── 11. SAVE ENRICHED CSV ────────────────────────────────────────────────
    csv_path = os.path.join(args.out, "reviews_enriched.csv")
    df.to_csv(csv_path, index=False)
    console.print(f"[dim]Enriched dataset saved → {csv_path}[/dim]")

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    console.print(Panel(
        f"[bold green]✓ Analysis complete![/bold green]\n\n"
        f"  📊 [cyan]{len(df):,}[/cyan] reviews / posts analysed\n"
        f"  🎮 [cyan]{df['game'].nunique()}[/cyan] games\n"
        f"  💚 Overall sentiment: [bold]{df['compound'].mean():+.3f}[/bold]\n"
        f"  🏆 Best rated: [bold yellow]{game_stats.iloc[0]['game']}[/bold yellow]\n"
        f"  📂 Output: [underline]{os.path.abspath(args.out)}[/underline]\n"
        f"  🌐 Open: [underline]{os.path.abspath(report_path)}[/underline]",
        title="⚔️  Done",
        border_style="green",
        expand=False,
    ))


if __name__ == "__main__":
    main()
