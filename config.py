"""
Soulslike Sentiment Analyzer - Configuration
"""

# ─── Steam App IDs ──────────────────────────────────────────────────────────
# genre values:
#   "soulslike"            — core soulslike mechanics (stamina, death penalty, etc.)
#   "soulslike_2d"         — 2D / metroidvania soulslike
#   "soulslike_shooter"    — soulslike with ranged / co-op focus
#   "action_rpg_adjacent"  — action RPG with some soulslike DNA
#   "action_rpg"           — action RPG, low soulslike similarity
SOULSLIKE_GAMES = {

    # ── Core Soulsborne (FromSoftware) ───────────────────────────────────────
    "Elden Ring":                   {"appid": 1245620,  "year": 2022, "genre": "soulslike"},
    "Dark Souls III":               {"appid": 374320,   "year": 2016, "genre": "soulslike"},
    "Dark Souls Remastered":        {"appid": 570940,   "year": 2018, "genre": "soulslike"},
    "Dark Souls II: SotFS":         {"appid": 335300,   "year": 2014, "genre": "soulslike"},
    "Sekiro: Shadows Die Twice":    {"appid": 814380,   "year": 2019, "genre": "soulslike"},

    # ── Soulslike (third-party, 3D) ──────────────────────────────────────────
    "Lies of P":                    {"appid": 1627720,  "year": 2023, "genre": "soulslike"},
    "Nioh: Complete Edition":       {"appid": 485510,   "year": 2017, "genre": "soulslike"},
    "Nioh 2 Complete":              {"appid": 1325200,  "year": 2021, "genre": "soulslike"},
    "Code Vein":                    {"appid": 678960,   "year": 2019, "genre": "soulslike"},
    "Mortal Shell":                 {"appid": 1168810,  "year": 2020, "genre": "soulslike"},
    "The Surge 2":                  {"appid": 1001510,  "year": 2019, "genre": "soulslike"},
    "Lords of the Fallen":          {"appid": 1501750,  "year": 2023, "genre": "soulslike"},
    "Lords of the Fallen (2014)":   {"appid": 257790,   "year": 2014, "genre": "soulslike"},
    "Wo Long: Fallen Dynasty":      {"appid": 2183250,  "year": 2023, "genre": "soulslike"},
    "Steelrising":                  {"appid": 1341290,  "year": 2022, "genre": "soulslike"},
    "Thymesia":                     {"appid": 1696350,  "year": 2022, "genre": "soulslike"},
    "Hellpoint":                    {"appid": 916730,   "year": 2020, "genre": "soulslike"},
    "Star Wars Jedi: Fallen Order":  {"appid": 1172380, "year": 2019, "genre": "soulslike"},

    # ── Soulslike Metroidvania / 2D ──────────────────────────────────────────
    "Hollow Knight":                {"appid": 367520,   "year": 2017, "genre": "soulslike_2d"},
    "Salt and Sanctuary":           {"appid": 343400,   "year": 2016, "genre": "soulslike_2d"},
    "Blasphemous":                  {"appid": 774361,   "year": 2019, "genre": "soulslike_2d"},
    "Blasphemous 2":                {"appid": 2114740,  "year": 2023, "genre": "soulslike_2d"},
    "Death's Door":                 {"appid": 1133240,  "year": 2021, "genre": "soulslike_2d"},
    "Tunic":                        {"appid": 553420,   "year": 2022, "genre": "soulslike_2d"},
    "Eldest Souls":                 {"appid": 1148530,  "year": 2021, "genre": "soulslike_2d"},

    # ── Soulslike with shooter / co-op ───────────────────────────────────────
    "Remnant: From the Ashes":      {"appid": 617290,   "year": 2019, "genre": "soulslike_shooter"},
    "Remnant II":                   {"appid": 1282100,  "year": 2023, "genre": "soulslike_shooter"},

    # ── Action RPG (soulslike-adjacent) ─────────────────────────────────────
    "NieR: Automata":               {"appid": 524220,   "year": 2017, "genre": "action_rpg_adjacent"},
    "Devil May Cry 5":              {"appid": 601150,   "year": 2019, "genre": "action_rpg_adjacent"},
    "Hades":                        {"appid": 1145360,  "year": 2020, "genre": "action_rpg_adjacent"},
    "Hades II":                     {"appid": 1145350,  "year": 2024, "genre": "action_rpg_adjacent"},
    "Dragon's Dogma 2":             {"appid": 2054970,  "year": 2024, "genre": "action_rpg_adjacent"},
    "Monster Hunter: World":        {"appid": 582010,   "year": 2018, "genre": "action_rpg_adjacent"},
    "Monster Hunter Rise":          {"appid": 1446780,  "year": 2022, "genre": "action_rpg_adjacent"},
    "Ghost of Tsushima":            {"appid": 2215430,  "year": 2024, "genre": "action_rpg_adjacent"},

    # ── Action RPG (low soulslike similarity) ────────────────────────────────
    "God of War":                   {"appid": 1593500,  "year": 2022, "genre": "action_rpg"},
    "The Witcher 3":                {"appid": 292030,   "year": 2015, "genre": "action_rpg"},
    "Horizon Zero Dawn":            {"appid": 1151640,  "year": 2020, "genre": "action_rpg"},
    "Cyberpunk 2077":               {"appid": 1091500,  "year": 2020, "genre": "action_rpg"},
    "Baldur's Gate 3":              {"appid": 1086940,  "year": 2023, "genre": "action_rpg"},
    "Control":                      {"appid": 870780,   "year": 2019, "genre": "action_rpg"},
    "Assassin's Creed Odyssey":     {"appid": 812140,   "year": 2018, "genre": "action_rpg"},
    "Mass Effect: Legendary Ed.":   {"appid": 1328670,  "year": 2021, "genre": "action_rpg"},
}

# ─── Reddit Subreddits ──────────────────────────────────────────────────────
SOULSLIKE_SUBREDDITS = [
    # ── Soulsborne / FromSoftware ────────────────────────────────────────────
    "Soulsborne",           # umbrella community for all FromSoft games
    "fromsoftware",         # developer-focused discussion
    "darksouls",            # DS1 / general Dark Souls
    "DarkSouls2",
    "DarkSouls3",
    "Eldenring",
    "Sekiro",

    # ── Third-party soulslikes ───────────────────────────────────────────────
    "LiesOfP",
    "Nioh",
    "codevein",
    "MortalShell",
    "LordsoftheFallen",
    "remnantgame",          # Remnant 1 & 2
    "WoLong",

    # ── Soulslike 2D / Metroidvania ──────────────────────────────────────────
    "HollowKnight",
    "Blasphemous",
    "tunicgame",

    # ── Action RPG adjacent ──────────────────────────────────────────────────
    "nier",                 # NieR: Automata
    "HadesTheGame",
    "Hades2",
    "MonsterHunterWorld",
    "MonsterHunter",
    "ghostoftsushima",
    "dragonsdogma",
    "DevilMayCry",

    # ── Action RPG (broader) ─────────────────────────────────────────────────
    "GodofWar",
    "witcher",
    "horizon",
    "cyberpunkgame",
    "BaldursGate3",
    "assassinscreed",

    # ── General / cross-genre ────────────────────────────────────────────────
    "patientgamers",        # retrospective reviews, great for older titles
    "JRPG",
    "actionrpg",
    "rpg",
    "Games",
]

REDDIT_SEARCH_TERMS = [
    # ── Soulslike sentiment / opinion ────────────────────────────────────────
    "best soulslike game",
    "soulslike tier list",
    "soulslike review",
    "soulslike difficulty",
    "soulslike beginner tips",
    "soulslike frustrating",
    "soulslike rewarding",
    "souls game ranking",
    "soulslike hidden gem",

    # ── Dark fantasy / atmosphere ────────────────────────────────────────────
    "dark fantasy game review",
    "dark fantasy atmosphere",
    "dark souls lore discussion",
    "elden ring lore",
    "best dark fantasy RPG",

    # ── Comparisons & recommendations ────────────────────────────────────────
    "soulslike vs action RPG",
    "dark souls vs elden ring",
    "is soulslike genre dying",
    "souls combat feel",
    "best action RPG 2023",
    "best action RPG 2024",
    "action RPG recommendation",

    # ── Specific themes that drive community discussion ───────────────────────
    "game difficulty debate",
    "punishing game worth it",
    "most satisfying combat",
    "open world action RPG",
    "boss fight design",
    "game that respects your time",
]

# ─── Scraping settings ──────────────────────────────────────────────────────
STEAM_REVIEWS_PER_GAME   = 200
REDDIT_POSTS_PER_SEARCH  = 50
STEAM_REVIEW_LANGUAGE    = "english"
REQUEST_DELAY_SECONDS    = 0.8

# ─── Analysis settings ──────────────────────────────────────────────────────
N_TOPICS           = 7
N_CLUSTERS         = 5
TOP_KEYWORDS       = 30
MIN_WORD_LENGTH    = 3
MAX_FEATURES_TFIDF = 3000

# ─── Output ─────────────────────────────────────────────────────────────────
OUTPUT_DIR = "output"
DATA_DIR   = "data"

# ─── Domain stop words ──────────────────────────────────────────────────────
CUSTOM_STOP_WORDS = {
    "game","games","play","playing","played","hours","time","like","just",
    "really","know","think","feel","good","great","one","get","got","also",
    "even","still","much","well","love","hate","want","need","make","made",
    "way","bit","thing","things","lot","sure","right","back","very","new",
    "first","come","came","look","looks","pretty","little","steam","review",
    "recommend","buy","purchase","hour","would","could","should","people",
    "when","with","this","that","have","been","from","they","them","their",
}

SENTIMENT_THRESHOLDS = {
    "very_positive":  0.5,
    "positive":       0.1,
    "neutral":       -0.1,
    "negative":      -0.5,
}