"""
HTML Report Generator — v2
Embeds Plotly charts DIRECTLY as div+script (no iframes, no base64).
Plotly.js loaded once from CDN → works in all browsers.
"""

import os
import re
import base64
from datetime import datetime
import pandas as pd


# ─── Extract div + chart-data script from a Plotly HTML file ─────────────────
def _extract_plotly(html_path: str) -> str:
    """Return the <div> + <script> pair for the chart, without the Plotly bundle."""
    if not html_path or not os.path.exists(html_path):
        return "<p style='color:#666;padding:20px;text-align:center'>Chart not available</p>"
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        # The div
        div_m = re.search(r'<div id="[^"]+" class="plotly-graph-div"[^>]*></div>', content)
        # All scripts; skip [0]=PlotlyConfig, [1]=bundle, [2]=chart data
        scripts = re.findall(r'<script[^>]*>(.*?)</script>', content, re.DOTALL)
        if not div_m or len(scripts) < 3:
            return "<p style='color:#666;padding:20px'>Parse error</p>"
        chart_script = scripts[2]
        return f'{div_m.group(0)}\n<script>{chart_script}</script>'
    except Exception as e:
        return f"<p style='color:#f55;padding:20px'>Error: {e}</p>"


def _img_tag(png_path: str) -> str:
    if not png_path or not os.path.exists(png_path):
        return ""
    with open(png_path, "rb") as f:
        enc = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{enc}" style="width:100%;display:block">'


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _badge(label: str) -> str:
    cls = {"Very Positive": "vp", "Positive": "p", "Neutral": "n",
           "Negative": "ng", "Very Negative": "vn"}.get(label, "n")
    return f'<span class="badge badge-{cls}">{label}</span>'


def _game_table(game_stats: pd.DataFrame) -> str:
    rows = ""
    for _, r in game_stats.iterrows():
        c = float(r.get("mean_compound", 0))
        if   c >=  0.5: label = "Very Positive"
        elif c >=  0.1: label = "Positive"
        elif c >= -0.1: label = "Neutral"
        elif c >= -0.5: label = "Negative"
        else:           label = "Very Negative"
        pct = r.get("pct_positive", 0) or 0
        pt  = r.get("mean_playtime_h", 0) or 0
        rows += (
            f"<tr>"
            f"<td><b>#{int(r.get('rank', 0))}</b></td>"
            f"<td>{r['game']}</td>"
            f"<td><b>{c:+.3f}</b></td>"
            f"<td>{_badge(label)}</td>"
            f"<td>{int(r.get('n_reviews', 0))}</td>"
            f"<td>{float(pct):.1%}</td>"
            f"<td>{float(pt):.1f}h</td>"
            f"</tr>"
        )
    return f"""
    <table>
      <thead><tr>
        <th>#</th><th>Game</th><th>Avg Sentiment</th><th>Label</th>
        <th>Reviews</th><th>% Positive</th><th>Avg Playtime</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def _topic_table(topics_df) -> str:
    if topics_df is None or (hasattr(topics_df, "empty") and topics_df.empty):
        return "<p style='color:#888'>No topic data</p>"
    rows = "".join(
        f"<tr><td>{r['topic_id']}</td><td><b>{r['label']}</b></td>"
        f"<td style='font-size:0.8rem;color:#aaa'>{r['top_words'][:80]}</td>"
        f"<td>{r['mean_weight']:.4f}</td></tr>"
        for _, r in topics_df.iterrows()
    )
    return f"""
    <table>
      <thead><tr><th>ID</th><th>Topic</th><th>Top Keywords</th><th>Weight</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


# ─── CSS ─────────────────────────────────────────────────────────────────────
_CSS = """
<style>
:root {
  --bg:#0f0f1a; --panel:#1a1a2e; --card:#16213e;
  --accent:#e94560; --green:#2ecc71; --text:#eaeaea;
  --muted:#888; --border:#2a2a4a;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',Arial,sans-serif}
header{background:var(--panel);border-bottom:2px solid var(--accent);
       padding:24px 40px;display:flex;justify-content:space-between;align-items:center}
header h1{font-size:1.8rem;color:var(--accent)}
header p{color:var(--muted);font-size:.9rem}
.container{max-width:1400px;margin:0 auto;padding:32px 24px}
.section-title{font-size:1.3rem;color:var(--accent);margin:36px 0 16px;
               border-left:4px solid var(--accent);padding-left:12px}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:32px}
.kpi-card{background:var(--card);border:1px solid var(--border);
          border-radius:10px;padding:20px;text-align:center}
.kpi-card .value{font-size:2.2rem;font-weight:700;color:var(--green)}
.kpi-card .label{color:var(--muted);font-size:.85rem;margin-top:6px}
.chart-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(600px,1fr));gap:20px;margin-bottom:24px}
.chart-wrap{background:var(--card);border:1px solid var(--border);
            border-radius:10px;padding:12px;overflow:hidden}
.chart-full{background:var(--card);border:1px solid var(--border);
            border-radius:10px;padding:12px;margin-bottom:24px;overflow:hidden}
table{width:100%;border-collapse:collapse;margin-bottom:24px}
th{background:var(--panel);color:var(--accent);padding:10px 14px;
   text-align:left;font-size:.85rem;border-bottom:1px solid var(--border)}
td{padding:9px 14px;font-size:.85rem;border-bottom:1px solid var(--border)}
tr:hover td{background:var(--panel)}
.badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:.75rem;font-weight:600}
.badge-vp{background:#1e6b43;color:#2ecc71}
.badge-p {background:#1a4028;color:#82e0aa}
.badge-n {background:#3a3a3a;color:#aaa}
.badge-ng{background:#6b3a1a;color:#e59866}
.badge-vn{background:#6b1a1a;color:#e74c3c}
footer{text-align:center;color:var(--muted);padding:24px;font-size:.8rem}
/* Plotly dark bg fix */
.plotly-graph-div { background: transparent !important; }
</style>
"""

# ─── Main generator ───────────────────────────────────────────────────────────
def generate_report(df: pd.DataFrame, game_stats: pd.DataFrame,
                    topics_df, chart_paths: dict,
                    out_dir: str) -> str:
    now       = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_reviews = len(df)
    n_games   = df["game"].nunique()
    avg_sent  = df["compound"].mean()
    pct_pos   = (df["sentiment_label"].isin(["Very Positive", "Positive"])).mean()
    best_game = game_stats.iloc[0]["game"] if not game_stats.empty else "—"

    kpis = [
        (f"{n_reviews:,}",  "Total Reviews"),
        (str(n_games),      "Games Analysed"),
        (f"{avg_sent:.3f}", "Overall Sentiment"),
        (f"{pct_pos:.1%}",  "Positive Reviews"),
        (best_game[:20],    "Highest Rated"),
    ]
    kpi_html = "".join(
        f'<div class="kpi-card"><div class="value">{v}</div>'
        f'<div class="label">{l}</div></div>'
        for v, l in kpis
    )

    def chart(key):
        return _extract_plotly(chart_paths.get(key, ""))

    def img(key):
        return _img_tag(chart_paths.get(key, ""))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Soulslike Sentiment Analyzer — Report</title>
{_CSS}
<!-- Plotly loaded ONCE from CDN -->
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
</head>
<body>

<header>
  <div>
    <h1>⚔️ Soulslike Sentiment Analyzer</h1>
    <p>Data Science &amp; NLP analysis of Steam Reviews + Reddit posts</p>
  </div>
  <div style="text-align:right;color:var(--muted);font-size:.85rem">Generated: {now}</div>
</header>

<div class="container">

  <div class="section-title">📊 Key Metrics</div>
  <div class="kpi-grid">{kpi_html}</div>

  <div class="section-title">🏆 Sentiment Ranking — All Games</div>
  {_game_table(game_stats)}

  <div class="section-title">📈 Sentiment Analysis Charts</div>
  <div class="chart-grid">
    <div class="chart-wrap">{chart('ranking')}</div>
    <div class="chart-wrap">{chart('distribution')}</div>
  </div>
  <div class="chart-grid">
    <div class="chart-wrap">{chart('histogram')}</div>
    <div class="chart-wrap">{chart('volume')}</div>
  </div>

  <div class="section-title">📅 Sentiment Over Time</div>
  <div class="chart-full">{chart('temporal')}</div>

  <div class="section-title">🕷️ Multi-Metric Radar</div>
  <div class="chart-full" style="max-width:800px">{chart('radar')}</div>

  <div class="section-title">🎮 Playtime vs Sentiment</div>
  <div class="chart-full">{chart('playtime')}</div>

  <div class="section-title">🧠 NMF Topic Modelling</div>
  <div class="chart-full">{chart('topics')}</div>
  {_topic_table(topics_df)}

  <div class="section-title">🔍 Review Clusters (ML Segmentation)</div>
  <div class="chart-full">{chart('clusters')}</div>

  <div class="section-title">☁️ TF-IDF Keyword Clouds</div>
  <div class="chart-full">{img('wordcloud')}</div>

  <div class="section-title">🌐 Steam vs Reddit Comparison</div>
  <div class="chart-full" style="max-width:900px">{chart('sources')}</div>

</div>
<footer>Soulslike Sentiment Analyzer · VADER · scikit-learn · Plotly · {now}</footer>
</body>
</html>"""

    path = os.path.join(out_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    size_mb = os.path.getsize(path) / 1e6
    print(f"\n[Report] Dashboard saved → {path}  ({size_mb:.1f} MB)")
    return path
