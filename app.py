import streamlit as st
import pandas as pd
import math
import os

st.set_page_config(page_title="NCAAB Totals Model (Public Data)", layout="wide")

# =========================
# CONFIG
# =========================

DATA_FILE = "team_stats.csv"
LEAGUE_AVG_RTG = 103.0
MIN_POSSESSIONS = 65
STD_TOTAL = 11.0

# =========================
# NORMALIZATION
# =========================

ALIASES = {
    "michigan st": "michigan state",
    "st marys": "saint marys",
    "st johns": "saint johns",
}

def normalize(name):
    name = (
        str(name)
        .lower()
        .replace(".", "")
        .replace("&", "and")
        .replace("'", "")
        .strip()
    )
    return ALIASES.get(name, name)

# =========================
# LOAD TEAM DATA
# =========================

@st.cache_data(ttl=3600)
def load_teams():
    if not os.path.exists(DATA_FILE):
        st.error("Public data file missing. GitHub Action has not run yet.")
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

# =========================
# CORE MODEL (PUBLIC DATA)
# =========================

def projected_total(home, away, T):
    h = T[home]
    a = T[away]

    possessions = max((h["tempo"] + a["tempo"]) / 2, MIN_POSSESSIONS)

    home_ppp = (h["off"] / LEAGUE_AVG_RTG) * (LEAGUE_AVG_RTG / a["def"])
    away_ppp = (a["off"] / LEAGUE_AVG_RTG) * (LEAGUE_AVG_RTG / h["def"])

    return round(possessions * (home_ppp + away_ppp), 1)

def prob_over(proj, line):
    z = (proj - line) / STD_TOTAL
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

# =========================
# UI
# =========================

st.title("🏀 NCAAB Predictive Totals Model (Public Data)")

TEAM = load_teams()

home = st.text_input("Home Team")
away = st.text_input("Away Team")
line = st.number_input("Market Total", value=140.5)

if st.button("Project Total"):
    h = normalize(home)
    a = normalize(away)

    if h not in TEAM or a not in TEAM:
        st.error("Team not found in public dataset")
    else:
        proj = projected_total(h, a, TEAM)
        prob = prob_over(proj, line)

        st.metric("Projected Total", proj)
        st.metric("Edge", round(proj - line, 1))
        st.metric("Over Probability", f"{prob*100:.1f}%")
