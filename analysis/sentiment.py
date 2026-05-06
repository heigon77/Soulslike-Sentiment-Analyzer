"""
Sentiment Analysis
Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) — optimised for
social media / user-generated text without needing a model download.
"""

import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─── VADER custom lexicon for gaming / soulslike domain ─────────────────────
GAMING_LEXICON = {
    # positive domain-specific
    "masterpiece": 3.5, "flawless": 3.2, "addictive": 2.5, "rewarding": 2.8,
    "challenging": 1.5, "epic": 2.8, "immersive": 2.5, "gorgeous": 2.8,
    "legendary": 3.0, "phenomenal": 3.2, "satisfying": 2.6, "crisp": 1.8,
    "polished": 2.2, "atmospheric": 2.0, "brutal": 1.2, "intense": 1.5,
    "unforgiving": 0.8, "punishing": 0.5,
    # negative domain-specific
    "grindy": -1.8, "repetitive": -1.5, "tedious": -2.0, "laggy": -2.2,
    "clunky": -2.0, "janky": -2.3, "bloated": -1.5, "frustrating": -1.8,
    "unfair": -2.0, "buggy": -2.5, "unbalanced": -1.5, "hollow": -1.2,
    "disappointing": -2.2, "mediocre": -1.5, "soulless": -2.8,
    # neutral-positive verbs common in reviews
    "git": 0.0, "gud": 0.5, "souls": 0.2,
}


def _clean_text(text: str) -> str:
    """Light cleaning: remove URLs, extra whitespace."""
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def label_sentiment(compound: float) -> str:
    if   compound >=  0.5:  return "Very Positive"
    elif compound >=  0.1:  return "Positive"
    elif compound >= -0.1:  return "Neutral"
    elif compound >= -0.5:  return "Negative"
    else:                   return "Very Negative"


def analyse_sentiment(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Adds sentiment columns to the DataFrame in-place and returns it.
    New columns: compound, pos, neu, neg, sentiment_label, sentiment_score
    """
    analyser = SentimentIntensityAnalyzer()
    # inject gaming lexicon
    analyser.lexicon.update(GAMING_LEXICON)

    df = df.copy()
    df[text_col] = df[text_col].fillna("").apply(_clean_text)

    scores = df[text_col].apply(lambda t: analyser.polarity_scores(t))
    df["compound"]         = scores.apply(lambda s: round(s["compound"], 4))
    df["pos"]              = scores.apply(lambda s: round(s["pos"], 4))
    df["neu"]              = scores.apply(lambda s: round(s["neu"], 4))
    df["neg"]              = scores.apply(lambda s: round(s["neg"], 4))
    df["sentiment_label"]  = df["compound"].apply(label_sentiment)
    df["sentiment_score"]  = df["compound"]   # alias for clarity

    return df


def compute_game_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment metrics per game.
    Returns a DataFrame sorted by mean compound score (best game first).
    """
    steam_df = df[df["source"] == "steam"].copy()

    agg = (
        steam_df.groupby("game")
        .agg(
            n_reviews        = ("compound", "count"),
            mean_compound    = ("compound", "mean"),
            median_compound  = ("compound", "median"),
            pct_positive     = ("voted_up", "mean"),
            mean_playtime_h  = ("playtime_h", "mean"),
            std_compound     = ("compound", "std"),
        )
        .reset_index()
    )

    # sentiment category counts
    label_counts = (
        steam_df.groupby(["game", "sentiment_label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["Very Positive","Positive","Neutral","Negative","Very Negative"]:
        if col not in label_counts.columns:
            label_counts[col] = 0

    agg = agg.merge(label_counts, on="game", how="left")

    # positivity ratio: (VP+P) / total
    total = agg["n_reviews"].clip(lower=1)
    agg["positivity_ratio"] = (
        agg.get("Very Positive", 0) + agg.get("Positive", 0)
    ) / total

    agg = agg.sort_values("mean_compound", ascending=False).reset_index(drop=True)
    agg["rank"] = agg.index + 1
    return agg
