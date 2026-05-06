"""
Steam Reviews Scraper
Uses the public Steam Web API — no API key required.
  Endpoint: https://store.steampowered.com/appreviews/{appid}?json=1
"""

import time
import requests
from datetime import datetime
from rich.console import Console
from rich.progress import track

console = Console()

STEAM_REVIEW_URL = "https://store.steampowered.com/appreviews/{appid}"
STEAM_APP_URL    = "https://store.steampowered.com/api/appdetails?appids={appid}&filters=basic"

HEADERS = {
    "User-Agent": "SoulsllikeAnalyzer/1.0 (academic sentiment research)",
    "Accept-Language": "en-US,en;q=0.9",
}


def _fetch_page(appid: int, cursor: str, count: int = 100, language: str = "english") -> dict:
    params = {
        "json":          1,
        "language":      language,
        "filter":        "recent",
        "review_type":   "all",
        "purchase_type": "all",
        "num_per_page":  count,
        "cursor":        cursor,
    }
    try:
        r = requests.get(
            STEAM_REVIEW_URL.format(appid=appid),
            params=params, headers=HEADERS, timeout=15
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        console.print(f"[red]Steam API error (appid={appid}): {e}[/red]")
        return {}


def fetch_reviews(game_name: str, appid: int, max_reviews: int = 200,
                  language: str = "english", delay: float = 0.8) -> list:
    """
    Collect up to `max_reviews` Steam reviews for a game.
    Returns list of normalised dicts ready for the DataFrame.
    """
    reviews   = []
    cursor    = "*"
    page_size = min(100, max_reviews)

    while len(reviews) < max_reviews:
        data = _fetch_page(appid, cursor, page_size, language)
        if not data or data.get("success") != 1:
            break

        raw = data.get("reviews", [])
        if not raw:
            break

        for r in raw:
            text = r.get("review", "").strip()
            if len(text) < 10:          # skip near-empty reviews
                continue
            reviews.append({
                "source":      "steam",
                "game":        game_name,
                "appid":       appid,
                "review_id":   str(r.get("recommendationid", "")),
                "text":        text,
                "voted_up":    r.get("voted_up", False),
                "playtime_h":  round(r.get("author", {}).get("playtime_forever", 0) / 60, 1),
                "votes_up":    r.get("votes_up", 0),
                "votes_funny": r.get("votes_funny", 0),
                "timestamp":   datetime.utcfromtimestamp(
                                   r.get("timestamp_created", 0)
                               ).strftime("%Y-%m-%d"),
            })

        new_cursor = data.get("cursor", "")
        if not new_cursor or new_cursor == cursor:
            break
        cursor = new_cursor
        time.sleep(delay)

    return reviews[:max_reviews]


def scrape_all_games(games: dict, max_per_game: int = 200,
                     language: str = "english", delay: float = 0.8) -> list:
    """Scrape reviews for every game defined in `games`."""
    all_reviews = []
    items = list(games.items())

    for game_name, meta in track(items, description="[cyan]Steam reviews"):
        appid   = meta["appid"]
        reviews = fetch_reviews(game_name, appid, max_per_game, language, delay)
        all_reviews.extend(reviews)
        console.print(f"  [green]✓[/green] {game_name:<35} {len(reviews):>4} reviews")
        time.sleep(delay)

    console.print(
        f"\n[bold green]Steam total:[/bold green] {len(all_reviews)} reviews collected"
    )
    return all_reviews
