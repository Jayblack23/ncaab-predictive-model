import streamlit as st
import pandas as pd
import math
import os

st.set_page_config(page_title="NCAAB Totals Model", layout="wide")

# ============================================================
# CONFIG
# ============================================================

LEAGUE_AVG_RTG = 103.0
MIN_POSSESSIONS = 65
STD_TOTAL = 11.0
DATA_FILE = "team_stats.csv"

# ============================================================
# TEAM NAME NORMALIZATION
# ============================================================

ALIASES = {
    "st marys": "saint marys",
    "st johns": "saint johns",
    "michigan st": "michigan state",
}

def normalize(name):
    return (
        str(name)
        .lower()
        .replace(".", "")
        .replace("&", "and")
        .replace("'", "")
        .strip()
    )

# ============================================================
# LOAD TEAM STATS
# ============================================================

@st.cache_data(ttl=3600)
def load_team_stats():
    if not os.path.exists(DATA_FILE):
        st.error("team_stats.csv not found. Run torvik_scraper.py")
        st.stop()

    df = pd.read_csv(DATA_FILE)

    teams = {}
    for _, r in df.iterrows():
        teams[normalize(r["Team"])] = {
            "tempo": float(r["Tempo"]),
            "off": float(r["AdjOE"]),
            "def": float(r["AdjDE"]),
        }

    return teams

# ============================================================
# CORE MODEL
# ============================================================

def projected_total(home, away, TEAM):
    h = TEAM[home]
    a = TEAM[away]

    possessions = max((h["tempo"] + a["tempo"]) / 2, MIN_POSSESSIONS)

    home_ppp = (h["off"] / LEAGUE_AVG_RTG) * (LEAGUE_AVG_RTG / a["def"])
    away_ppp = (a["off"] / LEAGUE_AVG_RTG) * (LEAGUE_AVG_RTG / h["def"])

    total = possessions * (home_ppp + away_ppp)
    return round(total, 1)

def prob_over(proj, line):
    z = (proj - line) / STD_TOTAL
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

# ============================================================
# UI
# ============================================================

st.title("🏀 NCAAB Predictive Totals Model (Auto-Updated)")

TEAM = load_team_stats()

home_team = st.text_input("Home Team")
away_team = st.text_input("Away Team")
market_total = st.number_input("Market Total", value=140.5)

if st.button("Project Total"):
    h = normalize(home_team)
    a = normalize(away_team)

    if h not in TEAM or a not in TEAM:
        st.error("Team not found in team_stats.csv")
    else:
        proj = projected_total(h, a, TEAM)
        prob = prob_over(proj, market_total)

        st.metric("Projected Total", proj)
        st.metric("Edge", round(proj - market_total, 1))
        st.metric("Over Probability", f"{prob*100:.1f}%")
