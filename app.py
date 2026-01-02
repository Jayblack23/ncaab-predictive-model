import streamlit as st
import pandas as pd
import requests
import math
import os

st.set_page_config(page_title="NCAAB Predictive Totals Model", layout="wide")

# ============================================================
# CONFIG
# ============================================================

ODDS_API_KEY = st.secrets["ODDS_API_KEY"]

CONF_THRESHOLD = 0.60
EDGE_THRESHOLD = 3.0

MIN_GAME_POSSESSIONS = 68
PACE_SMOOTHING = 0.30
HOME_BONUS = 1.8
LEAGUE_AVG_RTG = 100

DATA_FILE = "team_stats.csv"

# ============================================================
# TEAM NAME NORMALIZATION
# ============================================================

ALIASES = {
    "ole miss": "mississippi",
    "uconn": "connecticut",
    "st marys": "saint marys",
    "st johns": "saint johns",
    "unc": "north carolina",
    "uva": "virginia",
}

def normalize(name):
    if not name:
        return ""
    name = (
        name.lower()
        .replace("&", "and")
        .replace(".", "")
        .replace("'", "")
        .strip()
    )
    return ALIASES.get(name, name)

def resolve(name, teams):
    for t in teams:
        if name == t or name in t or t in name:
            return t
    return None

# ============================================================
# LOAD TEAM STATS (LOCAL CSV — SAFE)
# ============================================================

@st.cache_data(ttl=86400)
def load_team_stats():
    if not os.path.exists(DATA_FILE):
        st.error("team_stats.csv not found. Please add it to the project root.")
        st.stop()

    df = pd.read_csv(DATA_FILE)

    required = {"Team", "Tempo", "AdjOE", "AdjDE"}
    if not required.issubset(df.columns):
        st.error("team_stats.csv must contain: Team, Tempo, AdjOE, AdjDE")
        st.stop()

    teams = {}
    for _, r in df.iterrows():
        teams[normalize(r["Team"])] = {
            "poss": float(r["Tempo"]),
            "off": float(r["AdjOE"]),
            "def": float(r["AdjDE"]),
        }

    return teams

# ============================================================
# LOAD ODDS
# ============================================================

@st.cache_data(ttl=300)
def load_odds():
    url = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "totals",
        "oddsFormat": "american",
    }
    return requests.get(url, params=params).json()

# ============================================================
# CORE MODEL
# ============================================================

def projected_total(home, away, TEAM):
    h = TEAM[home]
    a = TEAM[away]

    raw_poss = (h["poss"] + a["poss"]) / 2
    possessions = raw_poss + PACE_SMOOTHING * (MIN_GAME_POSSESSIONS - raw_poss)

    home_ppp = (h["off"] / LEAGUE_AVG_RTG) * (LEAGUE_AVG_RTG / a["def"])
    away_ppp = (a["off"] / LEAGUE_AVG_RTG) * (LEAGUE_AVG_RTG / h["def"])

    total = possessions * (home_ppp + away_ppp)
    total += HOME_BONUS

    return round(total, 1)

def prob_over(proj, line):
    std = 11
    z = (proj - line) / std
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

# ============================================================
# UI
# ============================================================

st.title("🏀 NCAAB Predictive Totals Model")

TEAM = load_team_stats()
ODDS = load_odds()

rows = []

for g in ODDS:
    home_raw = normalize(g.get("home_team"))
    away_raw = normalize(g.get("away_team"))

    home = resolve(home_raw, TEAM)
    away = resolve(away_raw, TEAM)

    if not home or not away:
        continue

    totals = []
    for b in g.get("bookmakers", []):
        for m in b.get("markets", []):
            if m.get("key") == "totals":
                for o in m.get("outcomes", []):
                    if o.get("name") == "Over" and isinstance(o.get("point"), (int, float)):
                        totals.append(o["point"])

    if not totals:
        continue

    market_total = round(sum(totals) / len(totals), 1)
    proj = projected_total(home, away, TEAM)
    edge = round(proj - market_total, 2)
    prob = prob_over(proj, market_total)

    decision = "BET" if prob >= CONF_THRESHOLD and edge >= EDGE_THRESHOLD else "PASS"

    confidence = (
        "⭐⭐⭐" if prob >= 0.65 else
        "⭐⭐" if prob >= 0.60 else
        "⭐"
    )

    rows.append({
        "Game": f"{g['away_team']} @ {g['home_team']}",
        "Market Total": market_total,
        "Projected Total": proj,
        "Edge": edge,
        "Probability %": round(prob * 100, 1),
        "Confidence": confidence,
        "Decision": decision,
    })

if not rows:
    st.warning("No valid games found.")
else:
    df = pd.DataFrame(rows).sort_values("Edge", ascending=False)
    st.dataframe(df, use_container_width=True)
