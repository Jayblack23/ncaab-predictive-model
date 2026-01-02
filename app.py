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
# MODEL CONSTANTS
# ============================================================
LEAGUE_AVG_TEMPO = 68
LEAGUE_AVG_EFF = 102
TOTAL_STD_DEV = 11.5   # realistic NCAAB totals variance

# ============================================================
# SESSION STATE (ROI PERSISTENCE)
# ============================================================
if "bet_log" not in st.session_state:
    st.session_state.bet_log = []

# ============================================================
# SPORTSDATAIO — TEAM METRICS (2025 SEASON)
# ============================================================
@st.cache_data(ttl=86400)
def fetch_team_metrics():
    headers = {
        "Ocp-Apim-Subscription-Key": st.secrets["SPORTSDATAIO_API_KEY"]
    }

    # IMPORTANT: SportsDataIO uses SEASON END YEAR
    season = 2025
    url = f"https://api.sportsdata.io/v3/cbb/stats/json/TeamSeasonStats/{season}"

    r = requests.get(url, headers=headers)

    if r.status_code != 200:
        st.error(f"SportsDataIO Error {r.status_code}")
        st.write(r.text)
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

def project_game_total(home, away):
    A = teams[teams.Team == home].iloc[0]
    B = teams[teams.Team == away].iloc[0]

    # Average & normalize tempo
    tempo = (A.Tempo + B.Tempo) / 2
    tempo *= LEAGUE_AVG_TEMPO / tempo

    pts_a = expected_points(tempo, A.AdjOE, B.AdjDE)
    pts_b = expected_points(tempo, B.AdjOE, A.AdjDE)

    return pts_a + pts_b

def over_probability(projected, line):
    z = (projected - line) / TOTAL_STD_DEV
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

# ============================================================
# UI — INPUTS
# ============================================================
st.title("🏀 College Basketball Totals Model (SportsDataIO)")

team_list = sorted(teams.Team.unique())

c1, c2, c3 = st.columns(3)
with c1:
    home = st.selectbox("Home Team", team_list)
with c2:
    away = st.selectbox("Away Team", team_list)
with c3:
    line = st.number_input("Market Total", value=140.5, step=0.5)

min_prob = st.slider("Minimum Probability (%)", 50, 65, 55)
min_edge = st.slider("Minimum Edge (Points)", 1.0, 5.0, 2.0)

# ============================================================
# RUN MODEL
# ============================================================
if home != away:
    projected = round(project_game_total(home, away), 2)
    edge = round(projected - line, 2)
    prob = round(over_probability(projected, line) * 100, 2)

    decision = "BET" if prob >= min_prob and edge >= min_edge else "PASS"

    st.subheader("📊 Projection")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Projected Total", projected)
    m2.metric("Market Line", line)
    m3.metric("Edge", edge)
    m4.metric("Over Probability", f"{prob}%")
    m5.metric("Decision", decision)

# ============================================================
# BET LOGGING
# ============================================================
st.subheader("🧾 Log Bet Result")

result = st.radio("Game Outcome", ["Pending", "Win", "Loss"], horizontal=True)

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

roi = round((units / total_bets) * 100, 2) if total_bets else 0
win_pct = round((wins / total_bets) * 100, 2) if total_bets else 0

r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("Total Bets", total_bets)
r2.metric("Wins", wins)
r3.metric("Losses", losses)
r4.metric("Units", units)
r5.metric("ROI %", roi)

# ============================================================
# FOOTER
# ============================================================
st.caption(f"Updated {date.today()} | Data Source: SportsDataIO | Season: 2025")
