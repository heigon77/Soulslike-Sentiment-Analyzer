"""
collectors/steam_collector.py
Coleta reviews da Steam usando a API pública (sem autenticação).
Endpoint: https://store.steampowered.com/appreviews/{appid}?json=1
"""

from __future__ import annotations

import time
import logging
from typing import Any

import requests
import pandas as pd

from config import (
    STEAM_GAMES,
    STEAM_REVIEW_URL,
    STEAM_APP_DETAILS_URL,
    STEAM_REVIEWS_PER_PAGE,
    STEAM_MAX_PAGES,
    STEAM_LANGUAGE,
    STEAM_REVIEW_FILTER,
    STEAM_REQUEST_DELAY,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_get(url: str, params: dict, retries: int = 3) -> dict | None:
    """GET com retry e tratamento de erros."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.warning(f"[attempt {attempt+1}/{retries}] GET {url} falhou: {exc}")
            time.sleep(2 ** attempt)
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Detalhes do app (metacritic, developer, genre…)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_app_details(appid: int) -> dict:
    """Retorna metadados do jogo via API de detalhes da Steam."""
    data = _safe_get(STEAM_APP_DETAILS_URL, {"appids": appid, "cc": "us", "l": "en"})
    if not data:
        return {}
    info = data.get(str(appid), {})
    if not info.get("success"):
        return {}
    d = info.get("data", {})
    return {
        "appid":        appid,
        "name":         d.get("name", ""),
        "developer":    ", ".join(d.get("developers", [])),
        "publisher":    ", ".join(d.get("publishers", [])),
        "release_date": d.get("release_date", {}).get("date", ""),
        "genres":       ", ".join(g["description"] for g in d.get("genres", [])),
        "metacritic":   d.get("metacritic", {}).get("score", None),
        "price_usd":    d.get("price_overview", {}).get("final_formatted", "Free"),
        "platforms":    ", ".join(k for k, v in d.get("platforms", {}).items() if v),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Reviews de um único jogo
# ─────────────────────────────────────────────────────────────────────────────

def fetch_reviews_for_game(
    appid: int,
    game_name: str,
    max_pages: int = STEAM_MAX_PAGES,
) -> list[dict]:
    """
    Coleta até `max_pages * STEAM_REVIEWS_PER_PAGE` reviews de um jogo.
    Usa cursor-based pagination da API Steam.
    """
    reviews: list[dict] = []
    cursor = "*"
    url = STEAM_REVIEW_URL.format(appid=appid)

    logger.info(f"  ↳ {game_name} (appid={appid}) — coletando até {max_pages} páginas…")

    for page in range(max_pages):
        params: dict[str, Any] = {
            "json":         1,
            "language":     STEAM_LANGUAGE,
            "filter":       STEAM_REVIEW_FILTER,
            "num_per_page": STEAM_REVIEWS_PER_PAGE,
            "cursor":       cursor,
            "review_type":  "all",
            "purchase_type":"all",
        }
        data = _safe_get(url, params)

        if not data or data.get("success") != 1:
            logger.warning(f"    Página {page+1}: resposta inválida, abortando.")
            break

        batch = data.get("reviews", [])
        if not batch:
            logger.info(f"    Sem mais reviews na página {page+1}.")
            break

        for r in batch:
            author = r.get("author", {})
            reviews.append({
                # identidade
                "game_name":          game_name,
                "appid":              appid,
                "recommendationid":   r.get("recommendationid"),
                # conteúdo
                "review":             r.get("review", "").strip(),
                "voted_up":           r.get("voted_up", False),   # True = positivo
                "weighted_vote_score":float(r.get("weighted_vote_score", 0.0)),
                "votes_up":           r.get("votes_up", 0),
                "votes_funny":        r.get("votes_funny", 0),
                "comment_count":      r.get("comment_count", 0),
                # autor
                "steam_id":           author.get("steamid"),
                "playtime_forever_h": round(author.get("playtime_forever", 0) / 60, 1),
                "num_reviews":        author.get("num_reviews", 0),
                # datas
                "timestamp_created":  r.get("timestamp_created"),
                "timestamp_updated":  r.get("timestamp_updated"),
                # idioma
                "language":           r.get("language", ""),
                # recebeu o jogo de graça?
                "steam_purchase":     r.get("steam_purchase", False),
                "received_for_free":  r.get("received_for_free", False),
            })

        # avança cursor
        cursor = data.get("cursor", "")
        if not cursor:
            break

        logger.info(f"    Página {page+1}: +{len(batch)} reviews (total={len(reviews)})")
        time.sleep(STEAM_REQUEST_DELAY)

    return reviews


# ─────────────────────────────────────────────────────────────────────────────
#  Coleta todos os jogos configurados
# ─────────────────────────────────────────────────────────────────────────────

def collect_all_games(games: dict[str, int] | None = None) -> pd.DataFrame:
    """
    Coleta reviews de todos os jogos em STEAM_GAMES (ou no dict fornecido).
    Retorna DataFrame consolidado.
    """
    if games is None:
        games = STEAM_GAMES

    all_reviews: list[dict] = []
    details_list: list[dict] = []

    logger.info(f"=== Steam Collector: {len(games)} jogos ===")

    for game_name, appid in games.items():
        # Detalhes do app
        details = fetch_app_details(appid)
        if details:
            details_list.append(details)
            logger.info(f"  Detalhes OK: {game_name}")
        time.sleep(0.5)

        # Reviews
        reviews = fetch_reviews_for_game(appid, game_name)
        all_reviews.extend(reviews)
        logger.info(f"  Total acumulado: {len(all_reviews)} reviews\n")
        time.sleep(STEAM_REQUEST_DELAY)

    df = pd.DataFrame(all_reviews)

    # Pós-processamento básico
    if not df.empty:
        df["date"] = pd.to_datetime(df["timestamp_created"], unit="s", errors="coerce")
        df["year_month"] = df["date"].dt.to_period("M").astype(str)
        df["review_len"] = df["review"].str.len()
        df["word_count"] = df["review"].str.split().str.len()
        # Remove reviews vazias
        df = df[df["review_len"] > 10].reset_index(drop=True)

    logger.info(f"=== Coleta concluída: {len(df)} reviews válidas ===")
    return df, pd.DataFrame(details_list)
