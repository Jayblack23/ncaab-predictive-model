# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import pandas as pd
import requests
import math
from datetime import date

st.set_page_config(page_title="NCAAB Predictive Model", layout="wide")

# ============================================================
# SESSION STATE (ROI PERSISTENCE)
# ============================================================
if "bet_log" not in st.session_state:
    st.session_state.bet_log = []

if "bets" not in st.session_state:
    st.session_state.bets = []

# ============================================================
# CONSTANTS
# ============================================================
LEAGUE_AVG_TEMPO = 68
LEAGUE_AVG_EFF = 102
TOTAL_STD_DEV = 11.5
REGRESSION_WEIGHT = 0.15  # KenPom-style regression

# ============================================================
# LOAD TEAM METRICS (TEAMID-BASED — CRITICAL FIX)
# ============================================================
@st.cache_data(ttl=86400)
def load_teams():
    headers = {"Ocp-Apim-Subscription-Key": st.secrets["SPORTSDATAIO_API_KEY"]}
    url = "https://api.sportsdata.io/v3/cbb/stats/json/TeamSeasonStats/2025"
    r = requests.get(url, headers=headers)

    rows = []
    for t in r.json():
        rows.append({
            "TeamID": t["TeamID"],
            "Name": t["Name"],
            "Tempo": t.get("PossessionsPerGame", LEAGUE_AVG_TEMPO),
            "OE": t.get("OffensiveEfficiency", LEAGUE_AVG_EFF),
            "DE": t.get("DefensiveEfficiency", LEAGUE_AVG_EFF)
        })

    return pd.DataFrame(rows)

teams_df = load_teams()
TEAM_LOOKUP = teams_df.set_index("TeamID").to_dict("index")
TEAM_NAME_MAP = teams_df.set_index("TeamID")["Name"].to_dict()

# ============================================================
# LOAD TODAY'S GAMES
# ============================================================
@st.cache_data(ttl=600)
def load_games():
    headers = {"Ocp-Apim-Subscription-Key": st.secrets["SPORTSDATAIO_API_KEY"]}
    today = date.today().strftime("%Y-%m-%d")
    url = f"https://api.sportsdata.io/v3/cbb/scores/json/GamesByDate/{today}"
    r = requests.get(url, headers=headers)
    return r.json() if r.status_code == 200 else []

games = load_games()

# ============================================================
# MODEL CORE
# ============================================================
def expected_points(tempo, oe, de):
    return tempo * (oe / de)

def raw_projected_total(home_id, away_id):
    A = TEAM_LOOKUP[home_id]
    B = TEAM_LOOKUP[away_id]

    tempo = (A["Tempo"] + B["Tempo"]) / 2
    tempo *= LEAGUE_AVG_TEMPO / tempo

    return (
        expected_points(tempo, A["OE"], B["DE"]) +
        expected_points(tempo, B["OE"], A["DE"])
    )

def kenpom_fallback(proj):
    league_mean_total = LEAGUE_AVG_TEMPO * 2
    return proj * (1 - REGRESSION_WEIGHT) + league_mean_total * REGRESSION_WEIGHT

def prob_over(proj, line):
    z = (proj - line) / TOTAL_STD_DEV
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

# ============================================================
# UI CONTROLS
# ============================================================
st.title("🏀 College Basketball Predictive Betting Model")

min_prob = st.slider("Min Probability (%)", 50, 65, 55)
min_edge = st.slider("Min Edge (pts)", 1.0, 6.0, 2.0)

# ============================================================
# MANUAL MATCHUP TOOL (WORKING DROPDOWNS)
# ============================================================
st.subheader("🧪 Manual Matchup Projection")

team_ids = list(TEAM_NAME_MAP.keys())

c1, c2, c3 = st.columns(3)
with c1:
    away_id = st.selectbox("Away Team", team_ids, format_func=lambda x: TEAM_NAME_MAP[x])
with c2:
    home_id = st.selectbox("Home Team", team_ids, format_func=lambda x: TEAM_NAME_MAP[x])
with c3:
    manual_total = st.number_input("Market Total (optional)", 0.0, 200.0, 0.0)

if st.button("Project Matchup"):
    proj = raw_projected_total(home_id, away_id)
    fallback = kenpom_fallback(proj)
    line = manual_total if manual_total > 0 else fallback
    edge = proj - line
    prob = prob_over(proj, line)

    st.metric("Projected Total", round(proj, 2))
    st.metric("Comparison Line", round(line, 2))
    st.metric("Edge", round(edge, 2))
    st.metric("Over Probability %", round(prob * 100, 1))

# ============================================================
# TODAY'S SLATE (MODEL-FIRST + FALLBACK)
# ============================================================
st.subheader("📅 Today’s Games")

rows = []

for g in games:
    home_id = g.get("HomeTeamID")
    away_id = g.get("AwayTeamID")

    if home_id not in TEAM_LOOKUP or away_id not in TEAM_LOOKUP:
        continue

    proj = raw_projected_total(home_id, away_id)
    fallback = kenpom_fallback(proj)

    market_total = g.get("OverUnder")
    if market_total not in (None, 0):
        line = float(market_total)
        source = "Market"
    else:
        line = fallback
        source = "Fallback"

    edge = proj - line
    p_over = prob_over(proj, line)
    prob = max(p_over, 1 - p_over)

    decision = (
        "BET" if source == "Market"
        and prob * 100 >= min_prob
        and abs(edge) >= min_edge
        else "PASS"
    )

    rows.append({
        "Game": f"{TEAM_NAME_MAP[away_id]} @ {TEAM_NAME_MAP[home_id]}",
        "Line Used": round(line, 2),
        "Line Source": source,
        "Projected Total": round(proj, 2),
        "Edge": round(edge, 2),
        "Prob %": round(prob * 100, 1),
        "Decision": decision
    })

    if decision == "BET":
        st.session_state.bets.append(f"{TEAM_NAME_MAP[away_id]} @ {TEAM_NAME_MAP[home_id]}")

df = pd.DataFrame(rows)

st.dataframe(df.sort_values("Edge", ascending=False), use_container_width=True)

st.caption(f"Games displayed: {len(df)}")

# ============================================================
# ROI TRACKING
# ============================================================
st.subheader("📈 Performance Summary")

if st.session_state.bets:
    st.write("Grade Bets:")
    for i, bet in enumerate(st.session_state.bets):
        result = st.selectbox(
            bet,
            ["Pending", "Win", "Loss"],
            key=f"res_{i}"
        )
        if result == "Win":
            st.session_state.bet_log.append(1)
        elif result == "Loss":
            st.session_state.bet_log.append(-1)

total = len(st.session_state.bet_log)
wins = st.session_state.bet_log.count(1)
losses = st.session_state.bet_log.count(-1)
units = sum(st.session_state.bet_log)

roi = round((units / total) * 100, 2) if total else 0
win_pct = round((wins / total) * 100, 2) if total else 0

st.metric("Total Bets", total)
st.metric("Wins", wins)
st.metric("Losses", losses)
st.metric("Units", units)
st.metric("ROI %", roi)
st.metric("Win %", win_pct)
