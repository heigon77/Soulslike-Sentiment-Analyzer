"""
Demo Data Generator
Creates realistic synthetic Steam reviews and Reddit posts for demo/testing.
Reviews are seeded with real game characteristics and vocabulary.
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ─── Game personality profiles ────────────────────────────────────────────────
GAME_PROFILES = {
    "Elden Ring": {
        "sentiment_mean":  0.52, "sentiment_std": 0.28,
        "positivity_rate": 0.92, "n_reviews": 180,
        "year": 2022,
        "pos_vocab": [
            "open world","masterpiece","breathtaking","incredible","rewarding",
            "seamless","exploration","legendary","atmospheric","stunning",
            "majestic","freedom","vast","innovative","polished","amazing",
            "epic","immersive","phenomenal","satisfying","beautiful","perfect",
        ],
        "neg_vocab": [
            "invasions","performance","stuttering","frame drops","pvp balance",
            "reused assets","late game","repetitive bosses",
        ],
        "templates": [
            "This is a {adj} game. The {feature} is absolutely {quality}.",
            "Elden Ring is a true {adj}. {hours} hours in and I still find new {feature}.",
            "The open world is {quality}. Every corner hides something {adj2}.",
            "FromSoftware delivered a {adj} masterpiece. The bosses are {quality}.",
            "Best soulslike ever made. The {feature} alone makes it worth it.",
        ],
    },
    "Dark Souls III": {
        "sentiment_mean":  0.47, "sentiment_std": 0.30,
        "positivity_rate": 0.89, "n_reviews": 160,
        "year": 2016,
        "pos_vocab": [
            "fluid combat","boss design","tight controls","interconnected",
            "challenging","satisfying","atmosphere","lore","amazing","fun",
            "crisp","smooth","intense","memorable","polished","rewarding",
        ],
        "neg_vocab": [
            "linear","too easy for veterans","pvp meta","invasions","dlc pricing",
            "repetitive","not as good as ds1",
        ],
        "templates": [
            "DS3 has the most {adj} combat in the series. The {feature} is {quality}.",
            "Boss fights are {quality}. Spent {hours} hours and loved every death.",
            "The {feature} design is {adj}. Really {quality} experience overall.",
            "Not perfect but the {feature} makes up for everything. {adj} game.",
            "Still one of the best soulslikes. {feature} is {quality} as always.",
        ],
    },
    "Sekiro: Shadows Die Twice": {
        "sentiment_mean":  0.49, "sentiment_std": 0.32,
        "positivity_rate": 0.88, "n_reviews": 140,
        "year": 2019,
        "pos_vocab": [
            "posture system","parrying","sword combat","fluid","rewarding",
            "skill based","satisfying","beautiful","japan","shinobi","intense",
            "amazing","unique","challenging","masterpiece","perfect","crisp",
        ],
        "neg_vocab": [
            "no builds","no co-op","difficulty","no character customization",
            "very punishing","hard for beginners","no summons",
        ],
        "templates": [
            "The {feature} system is pure {adj}. Nothing else feels this {quality}.",
            "Learning the {feature} click is so {adj}. Most {quality} combat ever.",
            "Sekiro is brutally {adj2} but incredibly {quality}. {hours} hours of joy.",
            "Completely different from DS. The {feature} takes skill but is {adj}.",
            "Once it clicks, this is the most {adj} game FromSoft made. Pure {quality}.",
        ],
    },
    "Dark Souls Remastered": {
        "sentiment_mean":  0.42, "sentiment_std": 0.31,
        "positivity_rate": 0.84, "n_reviews": 130,
        "year": 2018,
        "pos_vocab": [
            "classic","interconnected world","lore","atmosphere","challenging",
            "rewarding","nostalgia","great","amazing","fun","bosses","pvp",
        ],
        "neg_vocab": [
            "overpriced","few changes","lazy remaster","blight town fps",
            "not much improved","hackers","invasions",
        ],
        "templates": [
            "The original classic, now {adj}. {feature} still holds up {quality}.",
            "Dark Souls 1 is still the {adj} of the genre. {feature} is legendary.",
            "Great remaster of a {adj} game. The {feature} is as {quality} as ever.",
            "Came back after years, still {adj}. The {feature} design is {quality}.",
        ],
    },
    "Lies of P": {
        "sentiment_mean":  0.44, "sentiment_std": 0.29,
        "positivity_rate": 0.85, "n_reviews": 150,
        "year": 2023,
        "pos_vocab": [
            "parry system","weapon assembly","dark atmosphere","pinocchio twist",
            "satisfying","polished","beautiful","innovative","challenging",
            "amazing","stunning","great","fun","rewarding","solid","impressive",
        ],
        "neg_vocab": [
            "too similar to sekiro","linear","short","dlc needed","no coop",
            "repetitive enemies","some bosses unfair",
        ],
        "templates": [
            "Incredibly {adj} soulslike. The {feature} system is {quality}.",
            "Best non-FromSoft soulslike. {feature} is {adj} and innovative.",
            "The {adj} aesthetic and {feature} make this stand out. {quality}.",
            "Surprised by how {adj} this is. {feature} rivals FromSoftware games.",
            "The {feature} mechanics are {adj}. Story is {quality} as well.",
        ],
    },
    "Hollow Knight": {
        "sentiment_mean":  0.60, "sentiment_std": 0.22,
        "positivity_rate": 0.95, "n_reviews": 170,
        "year": 2017,
        "pos_vocab": [
            "beautiful","art style","challenging","atmospheric","rewarding",
            "exploration","metroidvania","lore","indie gem","masterpiece",
            "tight controls","amazing","stunning","emotional","deep",
        ],
        "neg_vocab": [
            "navigation","map system","backtracking","some bosses too hard",
            "slow movement","confusing lore",
        ],
        "templates": [
            "Hollow Knight is a {adj} indie masterpiece. The {feature} is {quality}.",
            "Best {adj} game of its generation. {feature} is perfectly crafted.",
            "The {feature} and atmosphere are {adj}. {hours} hours and still going.",
            "Absolutely {adj}. The {feature} alone is worth the price. {quality}.",
            "Nothing compares to the {adj} world of Hallownest. {feature} is {quality}.",
        ],
    },
    "Nioh 2 Complete": {
        "sentiment_mean":  0.38, "sentiment_std": 0.33,
        "positivity_rate": 0.81, "n_reviews": 120,
        "year": 2021,
        "pos_vocab": [
            "deep build","yokai skills","fast combat","satisfying","rewarding",
            "challenging","amazing","fun","variety","customization","great",
        ],
        "neg_vocab": [
            "complicated","overwhelming","loot spam","not beginner friendly",
            "grindy","story confusing","too many mechanics",
        ],
        "templates": [
            "Deep and {adj} but very {adj2}. The {feature} system is {quality}.",
            "More complex than DS but {adj}. {hours} hours in the {feature}.",
            "If you like {adj} build crafting this is {quality}. The {feature} is deep.",
            "Demanding but {adj}. The {feature} has more depth than any other soulslike.",
        ],
    },
    "Code Vein": {
        "sentiment_mean":  0.24, "sentiment_std": 0.36,
        "positivity_rate": 0.72, "n_reviews": 110,
        "year": 2019,
        "pos_vocab": [
            "anime style","blood codes","coop","accessible","fun","companions",
            "unique aesthetic","satisfying","interesting","build variety",
        ],
        "neg_vocab": [
            "too easy","shallow","generic bosses","repetitive","anime tropes",
            "not as deep as ds","bland world","story cliche","mediocre",
        ],
        "templates": [
            "Fun anime soulslike but {adj2}. The {feature} system is {quality}.",
            "If you enjoy anime and souls games this is {adj}. {feature} is decent.",
            "Not perfect but {adj}. The {feature} makes it fun. {hours} hours.",
            "The coop {feature} is great. Solo can feel {adj2} at times though.",
        ],
    },
    "Mortal Shell": {
        "sentiment_mean":  0.28, "sentiment_std": 0.34,
        "positivity_rate": 0.74, "n_reviews": 100,
        "year": 2020,
        "pos_vocab": [
            "hardening mechanic","dark atmosphere","unique","affordable","fun",
            "challenging","interesting","atmospheric","compact","rewarding",
        ],
        "neg_vocab": [
            "short","repetitive","shallow","small","limited content",
            "not enough variety","feels unfinished","clunky at times",
        ],
        "templates": [
            "Interesting soulslike with a {adj} hardening mechanic. {quality} short.",
            "Good for the price. The {feature} is {adj} but content is {adj2}.",
            "Compact but {adj}. The {feature} adds a fresh twist. {quality}.",
            "Dark atmosphere is {adj}. Wish it had more {feature}. Overall {quality}.",
        ],
    },
    "Salt and Sanctuary": {
        "sentiment_mean":  0.35, "sentiment_std": 0.31,
        "positivity_rate": 0.79, "n_reviews": 90,
        "year": 2016,
        "pos_vocab": [
            "2d souls","challenging","rewarding","dark","art style","fun",
            "great value","deep","metroidvania","satisfying","hidden gem",
        ],
        "neg_vocab": [
            "janky","rough edges","coop bugs","derivative","obtuse",
            "confusing UI","some unfair deaths",
        ],
        "templates": [
            "2D Dark Souls done {adj}. The {feature} is {quality}.",
            "Surprisingly {adj} for an indie. {feature} nails the souls formula.",
            "Rough around the edges but {adj}. The {feature} keeps you hooked.",
        ],
    },
    "Blasphemous": {
        "sentiment_mean":  0.40, "sentiment_std": 0.29,
        "positivity_rate": 0.83, "n_reviews": 110,
        "year": 2019,
        "pos_vocab": [
            "art style","dark catholic","atmospheric","unique","beautiful",
            "lore","rewarding","challenging","stunning","memorable","crisp",
        ],
        "neg_vocab": [
            "backtracking","movement","no map markers","platforming sections",
            "some controls clunky","late game pacing",
        ],
        "templates": [
            "The {adj} Spanish Inquisition aesthetic is unlike anything else. {quality}.",
            "Stunning {feature} and dark {adj} lore. Some {adj2} sections though.",
            "Hidden gem. The {feature} is {adj} and unique. Story is {quality}.",
        ],
    },
    "The Surge 2": {
        "sentiment_mean":  0.22, "sentiment_std": 0.38,
        "positivity_rate": 0.68, "n_reviews": 90,
        "year": 2019,
        "pos_vocab": [
            "limb targeting","loot system","sci-fi","interesting","fun",
            "unique mechanics","okay","decent","satisfying","okay build",
        ],
        "neg_vocab": [
            "repetitive","mediocre","bland","not as good","boring enemies",
            "forgettable story","frustrating","clunky","jank","generic",
        ],
        "templates": [
            "The limb targeting {feature} is interesting. {adj} but {adj2}.",
            "Decent sci-fi soulslike. The {feature} is {adj} but story is {adj2}.",
            "Not the best soulslike but the {feature} is {adj}. {quality}.",
        ],
    },
    "Remnant: From the Ashes": {
        "sentiment_mean":  0.41, "sentiment_std": 0.29,
        "positivity_rate": 0.83, "n_reviews": 120,
        "year": 2019,
        "pos_vocab": [
            "coop","gun play","procedural","fun with friends","rewarding",
            "variety","randomized","good","challenging","satisfying","great",
        ],
        "neg_vocab": [
            "solo harder","repetitive runs","rng","confusing","solo frustrating",
            "some bad randomization","dated graphics",
        ],
        "templates": [
            "Soulslike with guns — surprisingly {adj}! {feature} with friends is {quality}.",
            "The {feature} coop system is {adj}. Randomized maps keep it {quality}.",
            "Unique {adj} take on soulslikes. The {feature} adds great variety.",
        ],
    },
    "Wo Long: Fallen Dynasty": {
        "sentiment_mean":  0.30, "sentiment_std": 0.35,
        "positivity_rate": 0.75, "n_reviews": 120,
        "year": 2023,
        "pos_vocab": [
            "deflect system","fast paced","fun","rewarding","morale system",
            "chinese setting","martial arts","satisfying","interesting","good",
        ],
        "neg_vocab": [
            "too easy","unfinished","repetitive","grindy","performance issues",
            "story confusing","bland enemies","disappointed","mediocre",
        ],
        "templates": [
            "The deflect {feature} is {adj} but the game feels {adj2} overall.",
            "Fun {adj} soulslike but disappointing compared to Nioh. {feature} is {quality}.",
            "The {adj} setting is great. {feature} system needs more polish. {quality}.",
        ],
    },
    "Dark Souls II: SotFS": {
        "sentiment_mean":  0.28, "sentiment_std": 0.40,
        "positivity_rate": 0.73, "n_reviews": 120,
        "year": 2014,
        "pos_vocab": [
            "build variety","pvp","covenants","dlc","challenging","rewarding",
            "fun","deep","long","content rich","great pvp","unique areas",
        ],
        "neg_vocab": [
            "worst in series","hitboxes","agility system","soul memory",
            "poor boss design","lost izalith vibes","not as tight","janky",
            "ADP","artificial difficulty","disappointing after ds1",
        ],
        "templates": [
            "Most controversial DS but still {adj}. The {feature} is {quality}.",
            "Divisive game. The {feature} is {adj} but {feature2} is {adj2}.",
            "More {adj} than people give it credit for. PVP is {quality}.",
            "DS2 gets a bad rep but the {feature} is {adj}. DLCs are {quality}.",
        ],
    },
}

# ─── Shared vocabulary ────────────────────────────────────────────────────────
ADJ_POS  = ["amazing","incredible","fantastic","brilliant","excellent",
            "phenomenal","outstanding","superb","flawless","gorgeous","crisp"]
ADJ_MED  = ["decent","okay","solid","alright","fair","average","passable"]
ADJ_NEG  = ["disappointing","mediocre","frustrating","tedious","janky",
            "clunky","repetitive","boring","underwhelming","unpolished"]

QUALITY_POS = ["top-notch","excellent","brilliant","world-class","incredible",
               "unmatched","perfect","outstanding"]
QUALITY_NEG = ["lacking","weak","poor","underwhelming","needs work","mediocre"]

FEATURES   = ["combat","level design","boss fights","world building","atmosphere",
              "controls","mechanics","story","lore","art direction","music",
              "enemy variety","exploration","build system","difficulty curve"]

HOURS_LIST  = [12,25,40,60,80,100,120,150,200,300]

REDDIT_TEMPLATES = [
    "Just beat {game} for the first time. The {feature} is absolutely {adj}. "
    "Took me {hours} hours but it was worth every death.",

    "Hot take: {game} has the best {feature} in the genre. Change my mind.",

    "After {hours} hours in {game} I finally understand what makes it {adj}. "
    "The {feature} is just on another level.",

    "Is {game} the most {adj} soulslike ever? The {feature} system alone puts it in GOAT territory.",

    "Comparing {game} to other soulslikes: the {feature} is clearly {adj} "
    "but the difficulty might turn off newcomers.",

    "I bounced off {game} three times before it clicked. Now {hours} hours in "
    "and I think the {feature} is genuinely {adj}.",

    "The {feature} in {game} is criminally underrated. This game is {adj} "
    "and more people need to talk about it.",

    "Finished {game} last night. The {feature} near the end is genuinely {adj}. "
    "What a {adj2} experience.",

    "Why does {game} get so much hate? The {feature} is {adj} "
    "and the atmosphere is {adj2}. Easily recommend.",

    "Finally started {game} after putting it off for months. "
    "The {feature} is {adj} and I keep coming back for more.",
]


def _fill_template(template: str, profile: dict, rng: np.random.Generator) -> str:
    """Fill a review template with game-specific vocabulary."""
    pos_v = profile["pos_vocab"]
    neg_v = profile["neg_vocab"]
    sentiment_mean = profile["sentiment_mean"]

    use_pos  = rng.random() < (sentiment_mean + 1) / 2
    adj_pool = ADJ_POS if use_pos else (ADJ_NEG if rng.random() < 0.4 else ADJ_MED)

    replacements = {
        "{adj}":    rng.choice(pos_v + ADJ_POS) if use_pos else rng.choice(ADJ_NEG),
        "{adj2}":   rng.choice(ADJ_MED + (ADJ_NEG if not use_pos else ADJ_POS)),
        "{quality}":rng.choice(QUALITY_POS if use_pos else QUALITY_NEG),
        "{feature}":rng.choice(FEATURES),
        "{feature2}":rng.choice(FEATURES),
        "{hours}":  str(rng.choice(HOURS_LIST)),
        "{game}":   "",  # will be replaced by caller
    }
    for k, v in replacements.items():
        template = template.replace(k, v)
    return template


def _random_date(year: int, rng: np.random.Generator) -> str:
    start = datetime(year, 1, 1)
    end   = datetime(min(year + 4, 2025), 12, 31)
    days  = (end - start).days
    return (start + timedelta(days=int(rng.integers(0, days)))).strftime("%Y-%m-%d")


def generate_demo_data(seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic reviews and Reddit posts.
    Sentiment distributions are calibrated per game from community knowledge.
    """
    rng     = np.random.default_rng(seed)
    records = []

    # ── Steam reviews ─────────────────────────────────────────────────────────
    for game, profile in GAME_PROFILES.items():
        n       = profile["n_reviews"]
        mean_   = profile["sentiment_mean"]
        std_    = profile["sentiment_std"]
        year    = profile["year"]

        for i in range(n):
            # sample a compound score, clipped to [-1, 1]
            compound = float(np.clip(rng.normal(mean_, std_), -1, 1))
            voted_up = rng.random() < profile["positivity_rate"]

            # pick template
            templates = profile.get("templates", REDDIT_TEMPLATES)
            tmpl      = str(rng.choice(templates))
            text      = _fill_template(tmpl, profile, rng)

            records.append({
                "source":      "steam",
                "game":        game,
                "appid":       None,
                "review_id":   f"s_{game[:4]}_{i}",
                "text":        text,
                "voted_up":    bool(voted_up),
                "playtime_h":  float(np.clip(rng.exponential(80), 1, 1500)),
                "votes_up":    int(rng.integers(0, 500)),
                "votes_funny": int(rng.integers(0, 50)),
                "timestamp":   _random_date(year, rng),
            })

    # ── Reddit posts ──────────────────────────────────────────────────────────
    subreddits = ["Soulsborne","Eldenring","DarkSouls3","Sekiro","LiesOfP"]
    game_list  = list(GAME_PROFILES.keys())

    for i in range(350):
        game    = str(rng.choice(game_list))
        profile = GAME_PROFILES[game]
        tmpl    = str(rng.choice(REDDIT_TEMPLATES))
        text    = _fill_template(
            tmpl.replace("{game}", game), profile, rng
        )
        year    = profile["year"]
        sub     = str(rng.choice(subreddits))
        score   = int(rng.integers(1, 5000))

        records.append({
            "source":      "reddit",
            "game":        game,
            "appid":       None,
            "review_id":   f"r_{i}",
            "text":        text,
            "voted_up":    score > 0,
            "playtime_h":  None,
            "votes_up":    score,
            "votes_funny": int(rng.integers(0, 200)),
            "timestamp":   _random_date(year, rng),
            "subreddit":   sub,
            "title":       text[:60],
        })

    df = pd.DataFrame(records)
    # fill missing columns
    for col in ["subreddit","title"]:
        if col not in df.columns:
            df[col] = ""
    df["subreddit"] = df["subreddit"].fillna("")
    df["title"]     = df["title"].fillna("")

    return df
