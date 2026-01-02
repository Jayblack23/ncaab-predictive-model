import streamlit as st
import pandas as pd
import math

st.set_page_config(page_title="NCAAB Predictive Totals Model", layout="wide")

# ============================================================
# CONFIG
# ============================================================

LEAGUE_AVG_RTG = 103.0
MIN_POSSESSIONS = 65
STD_TOTAL = 11.0

# ============================================================
# TEAM NAME NORMALIZATION
# ============================================================

ALIASES = {
    "st marys": "saint marys",
    "st johns": "saint johns",
    "nc state": "north carolina state",
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
# LOAD BART TORVIK CSV
# ============================================================

@st.cache_data(ttl=86400)
def load_team_stats(df):
    cols = {c.lower(): c for c in df.columns}

    def find_col(options):
        for o in options:
            if o.lower() in cols:
                return cols[o.lower()]
        return None

    team_col = find_col(["team"])
    tempo_col = find_col(["tempo", "pace"])
    off_col = find_col(["adjoe", "oe"])
    def_col = find_col(["adjde", "de"])

    if not all([team_col, tempo_col, off_col, def_col]):
        st.error("CSV must include Team, Tempo, AdjOE/OE, AdjDE/DE")
        st.stop()

    teams = {}
    for _, r in df.iterrows():
        teams[normalize(r[team_col])] = {
            "tempo": float(r[tempo_col]),
            "off": float(r[off_col]),
            "def": float(r[def_col]),
        }

    return teams

# ============================================================
# CORE TOTALS MODEL (FIXED)
# ============================================================

def projected_total(home, away, TEAM):
    h = TEAM[home]
    a = TEAM[away]

    # Possessions
    possessions = max((h["tempo"] + a["tempo"]) / 2, MIN_POSSESSIONS)

    # Offensive efficiency vs opponent defense
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

st.title("🏀 NCAAB Predictive Totals Model (Bart Torvik)")

uploaded = st.file_uploader(
    "Upload Bart Torvik CSV (Team, Tempo, AdjOE, AdjDE)",
    type=["csv"]
)

if not uploaded:
    st.info("Please upload a Bart Torvik CSV to continue.")
    st.stop()

df = pd.read_csv(uploaded)
TEAM = load_team_stats(df)

home_team = st.text_input("Home Team")
away_team = st.text_input("Away Team")
market_total = st.number_input("Market Total", value=140.5)

if st.button("Project Total"):
    h = normalize(home_team)
    a = normalize(away_team)

    if h not in TEAM or a not in TEAM:
        st.error("One or both teams not found in CSV")
    else:
        proj = projected_total(h, a, TEAM)
        prob = prob_over(proj, market_total)

        st.metric("Projected Total", proj)
        st.metric("Edge", round(proj - market_total, 1))
        st.metric("Over Probability", f"{prob*100:.1f}%")
