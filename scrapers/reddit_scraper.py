"""
Reddit Scraper (no auth required)
Uses Reddit's public JSON endpoints:
  https://www.reddit.com/r/{sub}/search.json
  https://www.reddit.com/r/{sub}/hot.json
"""

import time
import requests
from datetime import datetime
from rich.console import Console
from rich.progress import track

console = Console()

BASE_URL = "https://www.reddit.com"
HEADERS  = {
    "User-Agent": "SoulsllikeAnalyzer/1.0 (academic research; contact: research@example.com)",
}


def _get(url: str, params: dict = None) -> dict:
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        console.print(f"[red]Reddit error ({url}): {e}[/red]")
        return {}


def _parse_post(post: dict, source_tag: str = "") -> dict | None:
    d = post.get("data", {})
    text = (d.get("selftext", "") or "").strip()
    title = (d.get("title", "") or "").strip()
    combined = f"{title}. {text}".strip()
    if len(combined) < 15:
        return None
    return {
        "source":    "reddit",
        "game":      source_tag,
        "appid":     None,
        "review_id": d.get("id", ""),
        "text":      combined,
        "voted_up":  d.get("score", 0) >= 1,
        "playtime_h": None,
        "votes_up":  d.get("score", 0),
        "votes_funny": d.get("num_comments", 0),   # repurpose field
        "timestamp": datetime.utcfromtimestamp(
                         d.get("created_utc", 0)
                     ).strftime("%Y-%m-%d"),
        "subreddit": d.get("subreddit", ""),
        "title":     title,
    }


def fetch_subreddit_top(subreddit: str, limit: int = 50, delay: float = 0.8) -> list:
    """Fetch top posts from a subreddit."""
    posts   = []
    after   = None
    fetched = 0

    while fetched < limit:
        params = {"limit": min(50, limit - fetched), "t": "year"}
        if after:
            params["after"] = after
        url  = f"{BASE_URL}/r/{subreddit}/top.json"
        data = _get(url, params)

        children = data.get("data", {}).get("children", [])
        if not children:
            break

        for child in children:
            p = _parse_post(child, source_tag=subreddit)
            if p:
                posts.append(p)
        fetched += len(children)
        after = data.get("data", {}).get("after")
        if not after:
            break
        time.sleep(delay)

    return posts[:limit]


def search_subreddit(subreddit: str, query: str, limit: int = 50, delay: float = 0.8) -> list:
    """Search for posts matching a query in a subreddit."""
    posts  = []
    after  = None

    while len(posts) < limit:
        params = {
            "q":       query,
            "sort":    "relevance",
            "t":       "year",
            "limit":   min(25, limit - len(posts)),
            "type":    "link",
            "restrict_sr": 1,
        }
        if after:
            params["after"] = after
        url  = f"{BASE_URL}/r/{subreddit}/search.json"
        data = _get(url, params)

        children = data.get("data", {}).get("children", [])
        if not children:
            break

        for child in children:
            p = _parse_post(child, source_tag=f"{subreddit}:{query}")
            if p:
                posts.append(p)
        after = data.get("data", {}).get("after")
        if not after:
            break
        time.sleep(delay)

    return posts[:limit]


def scrape_reddit(subreddits: list, search_terms: list,
                  posts_per_search: int = 50, delay: float = 0.8) -> list:
    """
    Collect posts from multiple subreddits, via top-posts + search terms.
    """
    all_posts = []
    seen_ids  = set()

    # 1) Top posts per subreddit
    for sub in track(subreddits, description="[magenta]Reddit subreddits"):
        posts = fetch_subreddit_top(sub, limit=posts_per_search, delay=delay)
        for p in posts:
            if p["review_id"] not in seen_ids:
                seen_ids.add(p["review_id"])
                all_posts.append(p)
        console.print(f"  [green]✓[/green] r/{sub:<25} {len(posts):>4} posts")
        time.sleep(delay)

    # 2) Search terms across all subreddits
    for term in track(search_terms, description="[magenta]Reddit searches"):
        for sub in subreddits[:3]:   # limit to first 3 to avoid rate limiting
            posts = search_subreddit(sub, term, limit=20, delay=delay)
            for p in posts:
                if p["review_id"] not in seen_ids:
                    seen_ids.add(p["review_id"])
                    all_posts.append(p)
        time.sleep(delay * 2)

    console.print(
        f"\n[bold magenta]Reddit total:[/bold magenta] {len(all_posts)} posts collected"
    )
    return all_posts
