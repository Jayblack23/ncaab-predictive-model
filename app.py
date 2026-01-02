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
# SESSION STATE
# ============================================================
if "bet_log" not in st.session_state:
    st.session_state.bet_log = []

if "bets" not in st.session_state:
    st.session_state.bets = []

# ============================================================
# CONSTANTS
# ============================================================
TOTAL_STD_DEV = 11.5
REGRESSION_WEIGHT = 0.12

# ============================================================
# LOAD TEAM STATS (RAW DATA — NO FAKE METRICS)
# ============================================================
@st.cache_data(ttl=86400)
def load_teams():
    headers = {"Ocp-Apim-Subscription-Key": st.secrets["SPORTSDATAIO_API_KEY"]}
    url = "https://api.sportsdata.io/v3/cbb/stats/json/TeamSeasonStats/2025"
    r = requests.get(url, headers=headers)

    teams = []
    for t in r.json():
        if t["Games"] == 0:
            continue

        teams.append({
            "TeamID": t["TeamID"],
            "Name": t["Name"],
            "Games": t["Games"],
            "Points": t["Points"],
            "PointsAllowed": t["OpponentPoints"],
            "FGA": t["FieldGoalsAttempted"],
            "FTA": t["FreeThrowsAttempted"],
            "ORB": t["OffensiveRebounds"],
            "TOV": t["Turnovers"]
        })

    return pd.DataFrame(teams)

teams_df = load_teams()
TEAM = teams_df.set_index("TeamID").to_dict("index")
TEAM_NAME = teams_df.set_index("TeamID")["Name"].to_dict()

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
# MODEL CORE (DERIVED POSSESSIONS)
# ============================================================
def estimate_possessions(team):
    poss = (
        team["FGA"]
        - team["ORB"]
        + team["TOV"]
        + 0.44 * team["FTA"]
    ) / team["Games"]
    return max(poss, 60)  # floor to avoid extreme outliers

def team_ratings(team):
    poss = estimate_possessions(team)
    ortg = (team["Points"] / team["Games"]) / poss * 100
    drtg = (team["PointsAllowed"] / team["Games"]) / poss * 100
    return poss, ortg, drtg

def projected_total(home_id, away_id):
    home = TEAM[home_id]
    away = TEAM[away_id]

    home_poss, home_ortg, home_drtg = team_ratings(home)
    away_poss, away_ortg, away_drtg = team_ratings(away)

    possessions = (home_poss + away_poss) / 2

    home_pts = possessions * (home_ortg / away_drtg)
    away_pts = possessions * (away_ortg / home_drtg)

    return home_pts + away_pts

def fallback_total(proj):
    league_avg = 145
    return proj * (1 - REGRESSION_WEIGHT) + league_avg * REGRESSION_WEIGHT

def prob_over(proj, line):
    z = (proj - line) / TOTAL_STD_DEV
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

# ============================================================
# UI CONTROLS
# ============================================================
st.title("🏀 College Basketball Predictive Model")

min_prob = st.slider("Min Probability (%)", 50, 65, 55)
min_edge = st.slider("Min Edge (pts)", 1.0, 6.0, 2.0)

# ============================================================
# MANUAL MATCHUP TOOL
# ============================================================
st.subheader("🧪 Manual Matchup Projection")

team_ids = list(TEAM_NAME.keys())
c1, c2, c3 = st.columns(3)

with c1:
    away_id = st.selectbox("Away Team", team_ids, format_func=lambda x: TEAM_NAME[x])
with c2:
    home_id = st.selectbox("Home Team", team_ids, format_func=lambda x: TEAM_NAME[x])
with c3:
    manual_total = st.number_input("Market Total (optional)", 0.0, 200.0, 0.0)

if st.button("Project Matchup"):
    proj = projected_total(home_id, away_id)
    line = manual_total if manual_total > 0 else fallback_total(proj)
    edge = proj - line
    prob = prob_over(proj, line)

    st.metric("Projected Total", round(proj, 2))
    st.metric("Comparison Line", round(line, 2))
    st.metric("Edge", round(edge, 2))
    st.metric("Over Probability %", round(prob * 100, 1))

# ============================================================
# TODAY'S SLATE
# ============================================================
st.subheader("📅 Today’s Games")

rows = []

for g in games:
    home_id = g.get("HomeTeamID")
    away_id = g.get("AwayTeamID")

    if home_id not in TEAM or away_id not in TEAM:
        continue

    proj = projected_total(home_id, away_id)
    market = g.get("OverUnder")

    if market not in (None, 0):
        line = float(market)
        source = "Market"
    else:
        line = fallback_total(proj)
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
        "Game": f"{TEAM_NAME[away_id]} @ {TEAM_NAME[home_id]}",
        "Market/Fallback Line": round(line, 1),
        "Line Source": source,
        "Projected Total": round(proj, 1),
        "Edge": round(edge, 1),
        "Prob %": round(prob * 100, 1),
        "Decision": decision
    })

df = pd.DataFrame(rows)

st.dataframe(df.sort_values("Edge", ascending=False), use_container_width=True)
st.caption(f"Games displayed: {len(df)}")

# ============================================================
# ROI TRACKING
# ============================================================
st.subheader("📈 Performance Summary")

if st.session_state.bets:
    for i, bet in enumerate(st.session_state.bets):
        result = st.selectbox(bet, ["Pending", "Win", "Loss"], key=f"res_{i}")
        if result == "Win":
            st.session_state.bet_log.append(1)
        elif result == "Loss":
            st.session_state.bet_log.append(-1)

total = len(st.session_state.bet_log)
wins = st.session_state.bet_log.count(1)
losses = st.session_state.bet_log.count(-1)
units = sum(st.session_state.bet_log)

st.metric("Total Bets", total)
st.metric("Wins", wins)
st.metric("Losses", losses)
st.metric("Units", units)
st.metric("ROI %", round((units / total) * 100, 2) if total else 0)
