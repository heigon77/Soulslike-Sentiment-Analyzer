"""
collectors/reddit_collector.py
Coleta posts e comentários do Reddit via PRAW (API oficial).
Requer credenciais em config.py.
"""

from __future__ import annotations

import logging
import time
from typing import Iterator

import pandas as pd

from config import (
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    REDDIT_SUBREDDITS,
    REDDIT_SEARCH_QUERIES,
    REDDIT_POST_LIMIT,
    STEAM_REQUEST_DELAY,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Verificação de dependência
# ─────────────────────────────────────────────────────────────────────────────

def _get_reddit_client():
    try:
        import praw
    except ImportError:
        raise ImportError(
            "PRAW não instalado. Execute: pip install praw\n"
            "Também configure REDDIT_CLIENT_ID e REDDIT_CLIENT_SECRET em config.py"
        )

    if REDDIT_CLIENT_ID == "SEU_CLIENT_ID":
        raise ValueError(
            "Credenciais do Reddit não configuradas.\n"
            "1. Acesse https://www.reddit.com/prefs/apps\n"
            "2. Crie um app 'script'\n"
            "3. Preencha REDDIT_CLIENT_ID e REDDIT_CLIENT_SECRET em config.py"
        )

    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Coleta de posts
# ─────────────────────────────────────────────────────────────────────────────

def _extract_post(post, subreddit: str) -> dict:
    return {
        "source":        "reddit",
        "subreddit":     subreddit,
        "post_id":       post.id,
        "title":         post.title,
        "review":        post.selftext or post.title,  # usa titulo se sem corpo
        "score":         post.score,
        "upvote_ratio":  post.upvote_ratio,
        "num_comments":  post.num_comments,
        "url":           post.url,
        "created_utc":   post.created_utc,
        "author":        str(post.author) if post.author else "[deleted]",
        "is_nsfw":       post.over_18,
        "flair":         post.link_flair_text or "",
    }


def _extract_comments(post, max_comments: int = 50) -> list[dict]:
    """Extrai até max_comments comentários de um post."""
    comments = []
    try:
        post.comments.replace_more(limit=0)
        for c in list(post.comments.list())[:max_comments]:
            if hasattr(c, "body") and len(c.body) > 20:
                comments.append({
                    "source":       "reddit_comment",
                    "subreddit":    post.subreddit.display_name,
                    "post_id":      post.id,
                    "review":       c.body,
                    "score":        c.score,
                    "created_utc":  c.created_utc,
                    "author":       str(c.author) if c.author else "[deleted]",
                })
    except Exception as exc:
        logger.debug(f"Erro ao extrair comentários de {post.id}: {exc}")
    return comments


# ─────────────────────────────────────────────────────────────────────────────
#  Coleta por subreddit
# ─────────────────────────────────────────────────────────────────────────────

def fetch_subreddit_posts(
    reddit,
    subreddit_name: str,
    limit: int = REDDIT_POST_LIMIT,
    include_comments: bool = True,
) -> list[dict]:
    """Hot + Top posts de um subreddit."""
    rows = []
    try:
        sub = reddit.subreddit(subreddit_name)
        seen = set()

        for post in sub.hot(limit=limit // 2):
            if post.id in seen:
                continue
            seen.add(post.id)
            rows.append(_extract_post(post, subreddit_name))
            if include_comments:
                rows.extend(_extract_comments(post))
            time.sleep(0.3)

        for post in sub.top(time_filter="year", limit=limit // 2):
            if post.id in seen:
                continue
            seen.add(post.id)
            rows.append(_extract_post(post, subreddit_name))
            if include_comments:
                rows.extend(_extract_comments(post))
            time.sleep(0.3)

        logger.info(f"  r/{subreddit_name}: {len(rows)} itens")
    except Exception as exc:
        logger.warning(f"  r/{subreddit_name}: erro — {exc}")

    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  Coleta principal
# ─────────────────────────────────────────────────────────────────────────────

def collect_reddit_data(
    subreddits: list[str] | None = None,
    include_comments: bool = True,
) -> pd.DataFrame:
    """
    Coleta posts e comentários dos subreddits configurados.
    Retorna DataFrame pronto para análise.
    """
    if subreddits is None:
        subreddits = REDDIT_SUBREDDITS

    reddit = _get_reddit_client()
    all_rows: list[dict] = []

    logger.info(f"=== Reddit Collector: {len(subreddits)} subreddits ===")

    for sub in subreddits:
        rows = fetch_subreddit_posts(reddit, sub, include_comments=include_comments)
        all_rows.extend(rows)
        time.sleep(STEAM_REQUEST_DELAY)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
        df["year_month"] = df["date"].dt.to_period("M").astype(str)
        df["review_len"] = df["review"].str.len()
        df["word_count"] = df["review"].str.split().str.len()
        df = df[df["review_len"] > 20].reset_index(drop=True)

    logger.info(f"=== Reddit: {len(df)} itens coletados ===")
    return df
