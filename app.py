import streamlit as st
import pandas as pd
import math
import os

st.set_page_config(page_title="NCAAB Totals Model", layout="wide")

LEAGUE_AVG_RTG = 103.0
MIN_POSSESSIONS = 65
STD_TOTAL = 11.0
DATA_FILE = "team_stats.csv"

def normalize(name):
    return (
        str(name)
        .lower()
        .replace(".", "")
        .replace("&", "and")
        .replace("'", "")
        .strip()
    )

@st.cache_data(ttl=3600)
def load_teams():
    if not os.path.exists(DATA_FILE):
        st.error("team_stats.csv not found. GitHub Action has not run yet.")
        st.stop()

    df = pd.read_csv(DATA_FILE)
    teams = {}

    for _, r in df.iterrows():
        teams[normalize(r["Team"])] = {
            "tempo": r["Tempo"],
            "off": r["AdjOE"],
            "def": r["AdjDE"],
        }

    return teams

def projected_total(h, a, T):
    home = T[h]
    away = T[a]

    poss = max((home["tempo"] + away["tempo"]) / 2, MIN_POSSESSIONS)

    home_ppp = (home["off"] / LEAGUE_AVG_RTG) * (LEAGUE_AVG_RTG / away["def"])
    away_ppp = (away["off"] / LEAGUE_AVG_RTG) * (LEAGUE_AVG_RTG / home["def"])

    return round(poss * (home_ppp + away_ppp), 1)

def prob_over(proj, line):
    z = (proj - line) / STD_TOTAL
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

st.title("🏀 NCAAB Predictive Totals Model (Auto Updated)")

TEAM = load_teams()

home = st.text_input("Home Team")
away = st.text_input("Away Team")
line = st.number_input("Market Total", value=140.5)

if st.button("Project"):
    h, a = normalize(home), normalize(away)

    if h not in TEAM or a not in TEAM:
        st.error("Team not found in CSV")
    else:
        proj = projected_total(h, a, TEAM)
        prob = prob_over(proj, line)

        st.metric("Projected Total", proj)
        st.metric("Edge", round(proj - line, 1))
        st.metric("Over Probability", f"{prob*100:.1f}%")
