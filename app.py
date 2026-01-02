# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import pandas as pd
import requests
import math
from datetime import date

st.set_page_config(page_title="NCAAB Totals Model", layout="wide")

# ============================================================
# CONSTANTS (MODEL CONTROLS)
# ============================================================
LEAGUE_AVG_TEMPO = 68
LEAGUE_AVG_EFF = 102
TOTAL_STD_DEV = 11.5   # empirically reasonable for NCAAB totals

# ============================================================
# SESSION STATE (ROI PERSISTENCE)
# ============================================================
if "bet_log" not in st.session_state:
    st.session_state.bet_log = []

# ============================================================
# DATA PULL — SPORTSDATAIO (LEGAL)
# ============================================================
@st.cache_data(ttl=86400)
def fetch_team_metrics():
    headers = {
        "Ocp-Apim-Subscription-Key": st.secrets["SPORTSDATAIO_API_KEY"]
    }

    season = 2024
    url = f"https://api.sportsdata.io/v3/cbb/stats/json/TeamSeasonStats/{season}"

    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        st.error("SportsDataIO API unavailable")
        st.stop()

    data = r.json()

    rows = []
    for t in data:
        rows.append({
            "Team": t["Name"],
            "Tempo": t.get("PossessionsPerGame", LEAGUE_AVG_TEMPO),
            "AdjOE": t.get("OffensiveEfficiency", LEAGUE_AVG_EFF),
            "AdjDE": t.get("DefensiveEfficiency", LEAGUE_AVG_EFF)
        })

    return pd.DataFrame(rows)

teams = fetch_team_metrics()

# ============================================================
# MODEL FUNCTIONS
# ============================================================
def expected_points(tempo, off_eff, def_eff):
    return tempo * (off_eff / def_eff)

def game_projection(team_a, team_b):
    A = teams[teams.Team == team_a].iloc[0]
    B = teams[teams.Team == team_b].iloc[0]

    tempo = (A.Tempo + B.Tempo) / 2
    tempo *= LEAGUE_AVG_TEMPO / tempo  # normalization

    pts_a = expected_points(tempo, A.AdjOE, B.AdjDE)
    pts_b = expected_points(tempo, B.AdjOE, A.AdjDE)

    return pts_a + pts_b

def over_probability(total, line):
    z = (total - line) / TOTAL_STD_DEV
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

# ============================================================
# UI — INPUTS
# ============================================================
st.title("🏀 College Basketball Totals Model")

team_list = sorted(teams.Team.unique())
col1, col2, col3 = st.columns(3)

with col1:
    home = st.selectbox("Home Team", team_list)
with col2:
    away = st.selectbox("Away Team", team_list)
with col3:
    line = st.number_input("Total Line", value=140.5, step=0.5)

min_prob = st.slider("Minimum Probability %", 50, 65, 55)
min_edge = st.slider("Minimum Edge (pts)", 1.0, 5.0, 2.0)

# ============================================================
# RUN MODEL
# ============================================================
if home != away:
    projected_total = round(game_projection(home, away), 2)
    prob_over = round(over_probability(projected_total, line) * 100, 2)
    edge = round(projected_total - line, 2)

    decision = "BET" if prob_over >= min_prob and edge >= min_edge else "PASS"

    st.subheader("📊 Projection")
    st.metric("Projected Total", projected_total)
    st.metric("Market Line", line)
    st.metric("Edge (pts)", edge)
    st.metric("Over Probability", f"{prob_over}%")
    st.metric("Decision", decision)

# ============================================================
# BET TRACKING
# ============================================================
st.subheader("🧾 Log Result")

result = st.radio("Game Result", ["Pending", "Win", "Loss"], horizontal=True)

if st.button("Save Result"):
    if result == "Win":
        st.session_state.bet_log.append(1)
    elif result == "Loss":
        st.session_state.bet_log.append(-1)

# ============================================================
# ROI SUMMARY
# ============================================================
st.subheader("📈 Performance Summary")

total_bets = len(st.session_state.bet_log)
wins = st.session_state.bet_log.count(1)
losses = st.session_state.bet_log.count(-1)
units = sum(st.session_state.bet_log)

roi = round((units / total_bets) * 100, 2) if total_bets > 0 else 0
win_pct = round((wins / total_bets) * 100, 2) if total_bets > 0 else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Bets", total_bets)
c2.metric("Wins", wins)
c3.metric("Losses", losses)
c4.metric("Units", units)
c5.metric("ROI %", roi)

# ============================================================
# FOOTER
# ============================================================
st.caption(f"Last Updated: {date.today()} | Data: SportsDataIO")
