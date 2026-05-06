"""
analysis/sentiment_analyzer.py
Análise de sentimento com VADER + análise baseada em aspecto (ABSA).
"""

from __future__ import annotations

import re
import logging
from typing import Optional

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import SENTIMENT_THRESHOLDS, GAME_ASPECTS

logger = logging.getLogger(__name__)

_analyzer = SentimentIntensityAnalyzer()

# ─────────────────────────────────────────────────────────────────────────────
#  Pré-processamento
# ─────────────────────────────────────────────────────────────────────────────

# Gírias e jargões de gaming que VADER não conhece
GAMING_LEXICON: dict[str, float] = {
    # Positivo
    "masterpiece": 3.0, "goat": 2.5, "banger": 2.2, "peak": 2.0,
    "godlike": 2.8, "goated": 2.5, "based": 1.5, "bussin": 2.0,
    "underrated": 1.2, "satisfying": 2.0, "addicting": 1.8,
    "atmospheric": 1.5, "immersive": 1.8, "rewarding": 2.2,
    "polished": 1.5, "breathtaking": 2.5, "epic": 1.8,
    # Negativo
    "trash": -2.5, "garbage": -2.5, "broken": -2.0, "unplayable": -3.0,
    "dogwater": -2.5, "cope": -1.5, "soulless": -2.0, "cashgrab": -2.5,
    "jank": -1.8, "tedious": -1.5, "clunky": -1.8, "bloated": -1.5,
    "overrated": -1.5, "mediocre": -1.2, "frustrating": -1.8,
    # Neutro/contexto gaming
    "git gud": 0.5,  # na comunidade soulslike é levemente positivo
    "ng+": 0.3,
}

for word, score in GAMING_LEXICON.items():
    _analyzer.lexicon[word] = score


def preprocess_text(text: str) -> str:
    """Limpeza leve preservando pontuação (VADER precisa dela)."""
    text = re.sub(r"http\S+", " ", text)            # remove URLs
    text = re.sub(r"[^\x00-\x7F]+", " ", text)      # remove non-ASCII
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
#  Sentimento por texto
# ─────────────────────────────────────────────────────────────────────────────

def analyze_sentiment(text: str) -> dict:
    """Retorna scores VADER + label categórico."""
    if not isinstance(text, str) or not text.strip():
        return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0, "label": "neutral"}

    clean = preprocess_text(text)
    scores = _analyzer.polarity_scores(clean)
    compound = scores["compound"]

    t = SENTIMENT_THRESHOLDS
    if compound >= t["very_positive"]:
        label = "very_positive"
    elif compound >= t["positive"]:
        label = "positive"
    elif compound <= t["negative"]:
        label = "very_negative"
    elif compound < t["neutral_low"]:
        label = "negative"
    else:
        label = "neutral"

    return {
        "compound":  compound,
        "pos":       scores["pos"],
        "neu":       scores["neu"],
        "neg":       scores["neg"],
        "label":     label,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Análise Baseada em Aspecto (ABSA)
# ─────────────────────────────────────────────────────────────────────────────

def _get_sentence_window(words: list[str], keyword_idx: int, window: int = 10) -> str:
    """Retorna janela de `window` palavras ao redor do índice."""
    start = max(0, keyword_idx - window)
    end   = min(len(words), keyword_idx + window)
    return " ".join(words[start:end])


def analyze_aspects(text: str) -> dict[str, float | None]:
    """
    Para cada aspecto em GAME_ASPECTS, busca menções no texto e calcula
    o sentimento médio nos trechos relevantes.
    Retorna {aspecto: compound_score} ou None se não mencionado.
    """
    if not isinstance(text, str):
        return {a: None for a in GAME_ASPECTS}

    words = preprocess_text(text.lower()).split()
    results: dict[str, float | None] = {}

    for aspect, keywords in GAME_ASPECTS.items():
        scores_found = []
        for i, w in enumerate(words):
            if any(kw in w for kw in keywords):
                window_text = _get_sentence_window(words, i)
                s = _analyzer.polarity_scores(window_text)["compound"]
                scores_found.append(s)
        results[aspect] = float(np.mean(scores_found)) if scores_found else None

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Enriquecimento do DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def enrich_dataframe(df: pd.DataFrame, text_col: str = "review") -> pd.DataFrame:
    """
    Adiciona colunas de sentimento ao DataFrame.
    Inclui VADER scores + análise por aspecto.
    """
    logger.info(f"Calculando sentimento para {len(df)} registros…")

    # ── Sentimento geral ──────────────────────────────────────────────────────
    sentiment_rows = df[text_col].apply(analyze_sentiment)
    sentiment_df = pd.DataFrame(sentiment_rows.tolist())
    df = pd.concat([df, sentiment_df], axis=1)

    # ── Sentimento por aspecto ────────────────────────────────────────────────
    logger.info("Calculando análise por aspecto…")
    aspect_rows = df[text_col].apply(analyze_aspects)
    aspect_df = pd.DataFrame(aspect_rows.tolist()).add_prefix("aspect_")
    df = pd.concat([df, aspect_df], axis=1)

    # ── Features derivadas ────────────────────────────────────────────────────
    df["is_positive_review"] = df["voted_up"] if "voted_up" in df.columns else (df["compound"] > 0)
    df["sentiment_vs_thumb"]  = (
        df["compound"].apply(lambda c: "positive" if c > 0 else "negative")
        == df.get("voted_up", pd.Series(dtype=bool)).apply(lambda v: "positive" if v else "negative")
    ) if "voted_up" in df.columns else pd.Series([None] * len(df))

    logger.info("Enriquecimento concluído.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Resumo por jogo
# ─────────────────────────────────────────────────────────────────────────────

def sentiment_summary_by_game(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega métricas de sentimento por jogo."""
    agg = (
        df.groupby("game_name")
        .agg(
            total_reviews   =("compound", "count"),
            mean_compound   =("compound", "mean"),
            median_compound =("compound", "median"),
            std_compound    =("compound", "std"),
            pct_positive    =("label", lambda x: (x.isin(["positive","very_positive"])).mean()),
            pct_very_positive=("label", lambda x: (x == "very_positive").mean()),
            pct_neutral     =("label", lambda x: (x == "neutral").mean()),
            pct_negative    =("label", lambda x: (x.isin(["negative","very_negative"])).mean()),
            pct_very_negative=("label", lambda x: (x == "very_negative").mean()),
            mean_playtime_h =("playtime_forever_h", "mean") if "playtime_forever_h" in df.columns else ("compound", "count"),
            total_votes_up  =("votes_up", "sum") if "votes_up" in df.columns else ("compound", "count"),
        )
        .reset_index()
    )

    # Aspect scores médios por jogo
    aspect_cols = [c for c in df.columns if c.startswith("aspect_")]
    if aspect_cols:
        aspect_agg = df.groupby("game_name")[aspect_cols].mean().reset_index()
        agg = agg.merge(aspect_agg, on="game_name", how="left")

    agg["sentiment_rank"] = agg["mean_compound"].rank(ascending=False).astype(int)
    agg["popularity_rank"] = agg["total_reviews"].rank(ascending=False).astype(int)
    return agg.sort_values("sentiment_rank")
