# ⚔️ Soulslike Sentiment Analyzer

A **Data Science & NLP** tool to collect, analyse, and visualize community sentiment around soulslike games, using the **Steam Reviews API** and the **Reddit JSON API** — no API keys required.

---

## 📁 Project Structure

```
soulslike_analyzer/
├── main.py                        # Main CLI entry point
├── run_demo.py                    # Full pipeline with synthetic data (no internet needed)
├── config.py                      # Games, subreddits, and global parameters
├── requirements.txt
│
├── scrapers/
│   ├── steam_scraper.py           # Steam Reviews API (public, no auth)
│   ├── reddit_scraper.py          # Reddit JSON API (public, no auth)
│   └── demo_data.py               # Realistic synthetic data generator
│
├── analysis/
│   ├── sentiment.py               # VADER + custom gaming lexicon
│   └── ml_analysis.py             # TF-IDF, NMF, KMeans, temporal trend
│
└── visualization/
    ├── charts.py                  # 11 interactive Plotly charts
    └── report.py                  # Self-contained HTML dashboard
```

---

## 🚀 Installation

**Requirements:** Python 3.10+

```bash
# Clone or extract the project
cd soulslike_analyzer

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### Full run (Steam + Reddit)
```bash
python main.py
```

### Steam only
```bash
python main.py --steam-only
```

### Reddit only
```bash
python main.py --reddit-only
```

### Limit reviews per game
```bash
python main.py --max 100
```

### Quick test with fewer games
```bash
python main.py --games 3 --max 50
```

### Reload previously collected data (skip scraping)
```bash
python main.py --load
```

### Demo mode — no internet, synthetic data
```bash
python run_demo.py
```

> **Note:** The Steam and Reddit APIs block requests originating from cloud servers. Running locally on your own machine both work without any authentication.

---

## 🎮 Games Included

**Core Soulsborne — FromSoftware**

| Game | Year |
|------|------|
| Elden Ring | 2022 |
| Dark Souls III | 2016 |
| Dark Souls Remastered | 2018 |
| Dark Souls II: SotFS | 2014 |
| Sekiro: Shadows Die Twice | 2019 |

**Soulslike — Third-party (3D)**

| Game | Year |
|------|------|
| Lies of P | 2023 |
| Nioh: Complete Edition | 2017 |
| Nioh 2 Complete | 2021 |
| Code Vein | 2019 |
| Mortal Shell | 2020 |
| The Surge 2 | 2019 |
| Lords of the Fallen | 2023 |
| Lords of the Fallen (2014) | 2014 |
| Wo Long: Fallen Dynasty | 2023 |
| Steelrising | 2022 |
| Thymesia | 2022 |
| Hellpoint | 2020 |
| Star Wars Jedi: Fallen Order | 2019 |

**Soulslike — 2D / Metroidvania**

| Game | Year |
|------|------|
| Hollow Knight | 2017 |
| Salt and Sanctuary | 2016 |
| Blasphemous | 2019 |
| Blasphemous 2 | 2023 |
| Death's Door | 2021 |
| Tunic | 2022 |
| Eldest Souls | 2021 |

**Soulslike — Shooter / Co-op**

| Game | Year |
|------|------|
| Remnant: From the Ashes | 2019 |
| Remnant II | 2023 |

**Action RPG — Soulslike-adjacent**

| Game | Year |
|------|------|
| NieR: Automata | 2017 |
| Devil May Cry 5 | 2019 |
| Hades | 2020 |
| Hades II | 2024 |
| Dragon's Dogma 2 | 2024 |
| Monster Hunter: World | 2018 |
| Monster Hunter Rise | 2022 |
| Ghost of Tsushima | 2024 |

**Action RPG — Low soulslike similarity**

| Game | Year |
|------|------|
| God of War | 2022 |
| The Witcher 3 | 2015 |
| Horizon Zero Dawn | 2020 |
| Cyberpunk 2077 | 2020 |
| Baldur's Gate 3 | 2023 |
| Control | 2019 |
| Assassin's Creed Odyssey | 2018 |
| Mass Effect: Legendary Ed. | 2021 |

To add or remove games, edit the `SOULSLIKE_GAMES` dictionary in `config.py`.

---

## 📊 Generated Analyses

| # | Chart | Technique |
|---|-------|-----------|
| 01 | Sentiment ranking by game | Mean VADER compound score |
| 02 | Sentiment distribution by game | Stacked bar (Very Negative → Very Positive) |
| 03 | Most discussed games | Review volume × sentiment colour |
| 04 | Sentiment over time | Monthly trend (top 6 games) |
| 05 | Community topics | NMF with 7 semantic topics |
| 06 | Multi-metric radar | 5 normalised dimensions per game |
| 07 | Review clusters | KMeans + LSA (5 groups) |
| 08 | Score histogram | VADER compound distribution |
| 09 | Playtime vs Sentiment | Scatter with LOWESS trendline |
| 10 | Word clouds per game | TF-IDF keywords |
| 11 | Steam vs Reddit | Sentiment comparison across sources |

The final dashboard is generated at `output/report.html` — a single self-contained HTML file requiring no external dependencies beyond a connection to the Plotly CDN.

---

## 🧠 Machine Learning Pipeline

```
Reviews / Posts
      │
      ▼
  Text cleaning (regex, stop words)
      │
      ▼
  VADER Sentiment Analysis
  + Custom gaming lexicon
      │
      ├──► Compound score [-1, +1]
      ├──► Label (Very Negative → Very Positive)
      └──► Aggregated metrics per game
      │
      ▼
  TF-IDF Vectorization
      │
      ├──► Keywords per game (top 30)
      ├──► NMF Topic Modelling (7 topics)
      └──► KMeans Clustering (5 groups)
               │
               └──► LSA (TruncatedSVD) for dimensionality reduction
```

### Configurable parameters in `config.py`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `STEAM_REVIEWS_PER_GAME` | 200 | Reviews collected per game |
| `N_TOPICS` | 7 | Number of NMF topics |
| `N_CLUSTERS` | 5 | Number of KMeans clusters |
| `TOP_KEYWORDS` | 30 | Keywords per game (TF-IDF) |
| `REQUEST_DELAY_SECONDS` | 0.8 | Delay between requests (rate limiting) |

---

## 📦 Main Dependencies

| Library | Purpose |
|---------|---------|
| `vaderSentiment` | Sentiment analysis |
| `scikit-learn` | TF-IDF, NMF, KMeans, LSA |
| `pandas` / `numpy` | Data manipulation |
| `plotly` | Interactive charts |
| `wordcloud` + `matplotlib` | Word cloud images |
| `requests` | HTTP requests to APIs |
| `rich` | Formatted terminal output |

---

## 📂 Outputs

After running, the `output/` folder will contain:

```
output/
├── report.html               # Full dashboard (open in any browser)
├── reviews_enriched.csv      # Complete dataset with sentiment scores
├── 01_sentiment_ranking.html
├── 02_sentiment_distribution.html
├── ...
└── 10_wordclouds.png
```

---

## 🔧 Data Sources

### Steam Reviews API
- **Endpoint:** `https://store.steampowered.com/appreviews/{appid}?json=1`
- Public, no authentication required
- Returns review text, helpfulness votes, author playtime, and timestamp

### Reddit JSON API
- **Endpoint:** `https://www.reddit.com/r/{subreddit}/top.json`
- Public, no authentication required (User-Agent header only)
- Subreddits: Soulsborne, Eldenring, DarkSouls3, Sekiro, LiesOfP, darksouls, DarkSouls2

---

## 📄 License

MIT — free for academic and personal use.