#!/usr/bin/env python3
"""
Demo runner — uses synthetic data when APIs are blocked.
Produces the full analysis + HTML report.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from rich.console import Console
from rich.panel   import Panel

console = Console()

from scrapers.demo_data  import generate_demo_data
from analysis.sentiment  import analyse_sentiment, compute_game_stats
from analysis.ml_analysis import (
    extract_keywords_per_game, topic_modelling,
    cluster_reviews, cluster_summary, temporal_trend, radar_metrics,
)
from visualization.charts import generate_all
from visualization.report import generate_report
from config import (
    CUSTOM_STOP_WORDS, N_TOPICS, N_CLUSTERS, TOP_KEYWORDS,
    MIN_WORD_LENGTH, MAX_FEATURES_TFIDF, OUTPUT_DIR, DATA_DIR,
)
from main import print_sentiment_table, print_topics, print_cluster_summary

BANNER = """[bold red]⚔[/bold red][bold white]  SOULSLIKE SENTIMENT ANALYZER — DEMO MODE[/bold white]  [bold red]⚔[/bold red]
[dim]Synthetic data · Full analysis pipeline · Real algorithms[/dim]"""

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,   exist_ok=True)
    console.print(Panel(BANNER, expand=False, border_style="red"))

    # 1. Generate synthetic data
    console.rule("[bold cyan]Data Generation[/bold cyan]")
    console.print("Generating realistic synthetic soulslike reviews...")
    df_raw = generate_demo_data(seed=42)
    console.print(f"  [green]✓[/green] {len(df_raw):,} records generated "
                  f"([cyan]{(df_raw['source']=='steam').sum()} steam[/cyan] / "
                  f"[magenta]{(df_raw['source']=='reddit').sum()} reddit[/magenta])")
    df_raw.to_parquet(os.path.join(DATA_DIR, "demo_reviews.parquet"), index=False)

    # 2. Sentiment analysis
    console.rule("[bold green]Sentiment Analysis (VADER + Gaming Lexicon)[/bold green]")
    df = analyse_sentiment(df_raw, text_col="text")

    # 3. Game stats + ranking
    game_stats = compute_game_stats(df)
    print_sentiment_table(game_stats)

    # 4. TF-IDF
    console.rule("[bold yellow]TF-IDF Keyword Extraction[/bold yellow]")
    keywords_per_game = extract_keywords_per_game(
        df, CUSTOM_STOP_WORDS, top_n=TOP_KEYWORDS, min_word_len=MIN_WORD_LENGTH
    )
    for game, kws in list(keywords_per_game.items())[:4]:
        top5 = ", ".join(w for w, _ in kws[:5])
        console.print(f"  [cyan]{game}[/cyan]: {top5}")

    # 5. NMF Topics
    console.rule("[bold yellow]NMF Topic Modelling[/bold yellow]")
    topics_df, doc_topics, _, _ = topic_modelling(
        df, CUSTOM_STOP_WORDS, n_topics=N_TOPICS, max_features=MAX_FEATURES_TFIDF
    )
    df["topic_id"] = doc_topics.values
    print_topics(topics_df)

    # 6. KMeans clustering
    console.rule("[bold yellow]KMeans Clustering[/bold yellow]")
    df = cluster_reviews(df, CUSTOM_STOP_WORDS, n_clusters=N_CLUSTERS)
    cs = cluster_summary(df)
    print_cluster_summary(cs)

    # 7. Temporal
    console.rule("[bold yellow]Temporal Analysis[/bold yellow]")
    trend_df = temporal_trend(df)

    # 8. Radar
    radar_df = radar_metrics(df, game_stats)

    # 9. All charts
    console.rule("[bold green]Generating Visualizations[/bold green]")
    chart_paths = generate_all(
        df, game_stats, trend_df, topics_df, doc_topics,
        keywords_per_game, radar_df, OUTPUT_DIR
    )

    # 10. HTML report
    console.rule("[bold green]Building HTML Dashboard[/bold green]")
    report_path = generate_report(df, game_stats, topics_df, chart_paths, OUTPUT_DIR)

    # 11. CSV
    csv_path = os.path.join(OUTPUT_DIR, "reviews_enriched.csv")
    df.to_csv(csv_path, index=False)

    console.print(Panel(
        f"[bold green]✓ Full pipeline complete![/bold green]\n\n"
        f"  📊 [cyan]{len(df):,}[/cyan] reviews analysed\n"
        f"  🎮 [cyan]{df['game'].nunique()}[/cyan] soulslike games\n"
        f"  💚 Overall sentiment: [bold]{df['compound'].mean():+.3f}[/bold]\n"
        f"  🏆 Best rated: [bold yellow]{game_stats.iloc[0]['game']}[/bold yellow]\n"
        f"  📂 Charts: [underline]{os.path.abspath(OUTPUT_DIR)}[/underline]\n"
        f"  🌐 Report: [underline]{os.path.abspath(report_path)}[/underline]",
        title="⚔️  Done",
        border_style="green",
        expand=False,
    ))

if __name__ == "__main__":
    main()
