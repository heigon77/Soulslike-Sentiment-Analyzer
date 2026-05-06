"""
ML Analysis
  - TF-IDF keyword extraction per game
  - NMF topic modelling across all reviews
  - KMeans review clustering
  - Temporal sentiment trend
  - Radar metrics per game
"""

import re
import numpy  as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition           import NMF, TruncatedSVD
from sklearn.cluster                 import KMeans
from sklearn.preprocessing           import normalize
from sklearn.metrics                 import silhouette_score

_RE_PUNC   = re.compile(r"[^a-zA-Z\s]")
_RE_SPACES = re.compile(r"\s+")

def _preprocess(text: str) -> str:
    text = _RE_PUNC.sub(" ", text.lower())
    return _RE_SPACES.sub(" ", text).strip()


def build_stop_words(custom: set) -> list:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return list(ENGLISH_STOP_WORDS | custom)


# ─── TF-IDF keyword extraction ──────────────────────────────────────────────
def extract_keywords_per_game(df, custom_stops, top_n=20, min_word_len=3):
    stop_words = build_stop_words(custom_stops)
    steam_df   = df[df["source"] == "steam"].copy()
    result     = {}
    for game, group in steam_df.groupby("game"):
        docs = group["text"].apply(_preprocess).tolist()
        if len(docs) < 3:
            continue
        try:
            vec    = TfidfVectorizer(
                stop_words=stop_words, max_features=500, min_df=2,
                token_pattern=rf"\b[a-zA-Z]{{{min_word_len},}}\b",
            )
            matrix = vec.fit_transform(docs)
            scores = np.asarray(matrix.mean(axis=0)).flatten()
            words  = vec.get_feature_names_out()
            top_idx = scores.argsort()[::-1][:top_n]
            result[game] = [(words[i], float(scores[i])) for i in top_idx]
        except Exception:
            result[game] = []
    return result


# ─── NMF Topic Modelling ─────────────────────────────────────────────────────
TOPIC_LABELS = {
    0: "Difficulty & Learning Curve",
    1: "Combat & Boss Fights",
    2: "Story & Lore",
    3: "World Design & Exploration",
    4: "Technical Performance",
    5: "Multiplayer & Community",
    6: "Replayability & Endgame",
}

def topic_modelling(df, custom_stops, n_topics=7, max_features=3000, top_words=12):
    stop_words = build_stop_words(custom_stops)
    docs       = df["text"].apply(_preprocess).tolist()
    vec = TfidfVectorizer(
        stop_words=stop_words, max_features=max_features,
        min_df=3, ngram_range=(1, 2),
    )
    X   = vec.fit_transform(docs)
    nmf = NMF(n_components=n_topics, random_state=42, max_iter=300)
    W   = nmf.fit_transform(X)
    H   = nmf.components_
    feature_names = vec.get_feature_names_out()
    topics = []
    for idx, component in enumerate(H):
        top_idx   = component.argsort()[::-1][:top_words]
        top_words_ = [feature_names[i] for i in top_idx]
        weight     = float(W[:, idx].mean())
        topics.append({
            "topic_id":    idx,
            "label":       TOPIC_LABELS.get(idx, f"Topic {idx}"),
            "top_words":   ", ".join(top_words_),
            "mean_weight": round(weight, 4),
        })
    topics_df  = pd.DataFrame(topics)
    doc_topics = pd.Series(W.argmax(axis=1), name="topic_id")
    return topics_df, doc_topics, vec, nmf


# ─── KMeans Clustering ───────────────────────────────────────────────────────
def cluster_reviews(df, custom_stops, n_clusters=5):
    stop_words = build_stop_words(custom_stops)
    docs = df["text"].apply(_preprocess).tolist()
    vec  = TfidfVectorizer(stop_words=stop_words, max_features=2000, min_df=3)
    X    = vec.fit_transform(docs)
    svd  = TruncatedSVD(n_components=min(100, X.shape[1]-1), random_state=42)
    X_r  = normalize(svd.fit_transform(X))
    km   = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X_r)
    df = df.copy()
    df["cluster"] = labels
    try:
        sil = silhouette_score(X_r, labels, sample_size=min(3000, len(labels)))
        print(f"  Silhouette score: {sil:.3f}")
    except Exception:
        pass
    return df


def cluster_summary(df):
    rows = []
    for c, grp in df.groupby("cluster"):
        dominant = grp["game"].value_counts().idxmax() if "game" in grp else "—"
        rows.append({
            "cluster":        int(c),
            "size":           len(grp),
            "mean_sentiment": round(grp["compound"].mean(), 3),
            "dominant_game":  dominant,
            "pct_positive":   round(
                (grp["sentiment_label"].isin(["Very Positive","Positive"])).mean()*100, 1
            ),
        })
    return pd.DataFrame(rows).sort_values("mean_sentiment", ascending=False)


# ─── Temporal Trend ──────────────────────────────────────────────────────────
def temporal_trend(df, freq="M"):
    steam_df = df[df["source"] == "steam"].copy()
    steam_df["month"] = pd.to_datetime(steam_df["timestamp"], errors="coerce").dt.to_period(freq)
    top_games = (
        steam_df.groupby("game")["compound"].count().nlargest(6).index.tolist()
    )
    filtered = steam_df[steam_df["game"].isin(top_games)]
    trend = (
        filtered.groupby(["month","game"])["compound"]
        .mean().round(3).unstack("game")
    )
    trend.index = trend.index.astype(str)
    return trend.ffill().fillna(0)


# ─── Radar metrics ───────────────────────────────────────────────────────────
def radar_metrics(df, game_stats):
    gs = game_stats.copy()
    def norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn) if mx != mn else s * 0
    pr_col = "positivity_ratio" if "positivity_ratio" in gs.columns else "mean_compound"
    metrics = pd.DataFrame({
        "game":          gs["game"],
        "Sentiment":     norm(gs["mean_compound"]),
        "Positivity":    norm(gs[pr_col]),
        "Review Volume": norm(gs["n_reviews"]),
        "Avg Playtime":  norm(gs["mean_playtime_h"].fillna(0)),
        "Consistency":   1 - norm(gs["std_compound"].fillna(0)),
    })
    return metrics
