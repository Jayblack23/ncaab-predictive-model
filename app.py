# # ============================================================
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

if "graded_bets" not in st.session_state:
    st.session_state.graded_bets = set()

# ============================================================
# CONSTANTS
# ============================================================
TOTAL_STD_DEV = 11.5
REGRESSION_WEIGHT = 0.12
LEAGUE_AVG_TOTAL = 145

# Floors (CRITICAL)
MIN_POSSESSIONS = 60
MIN_EFFICIENCY = 80

# ============================================================
# LOAD TEAM STATS (CORRECT SCHEMA)
# ============================================================
@st.cache_data(ttl=86400)
def load_teams():
    headers = {"Ocp-Apim-Subscription-Key": st.secrets["SPORTSDATAIO_API_KEY"]}
    url = "https://api.sportsdata.io/v3/cbb/stats/json/TeamSeasonStats/2025"
    r = requests.get(url, headers=headers)

    teams = []
    for t in r.json():
        games = t.get("Games", 0)
        if games == 0:
            continue

        teams.append({
            "TeamID": t["TeamID"],
            "Name": t["Name"],
            "Games": games,
            "Points": t.get("Points", 0),
            "PointsAllowed": t.get("OpponentPointsPerGame", 0) * games,
            "FGA": t.get("FieldGoalsAttempted", 0),
            "FTA": t.get("FreeThrowsAttempted", 0),
            "ORB": t.get("OffensiveRebounds", 0),
            "TOV": t.get("Turnovers", 0),
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
# MODEL CORE (ZERO-SAFE)
# ============================================================
def estimate_possessions(team):
    poss = (
        team["FGA"]
        - team["ORB"]
        + team["TOV"]
        + 0.44 * team["FTA"]
    ) / team["Games"]

    return max(poss, MIN_POSSESSIONS)

def team_ratings(team):
    poss = estimate_possessions(team)

    ortg = (team["Points"] / team["Games"]) / poss * 100
    drtg = (team["PointsAllowed"] / team["Games"]) / poss * 100

    # 🔒 FLOORS (THIS FIXES YOUR ERROR)
    ortg = max(ortg, MIN_EFFICIENCY)
    drtg = max(drtg, MIN_EFFICIENCY)

    return poss, ortg, drtg

def projected_total(home_id, away_id):
    home = TEAM[home_id]
    away = TEAM[away_id]

    h_poss, h_ortg, h_drtg = team_ratings(home)
    a_poss, a_ortg, a_drtg = team_ratings(away)

    possessions = (h_poss + a_poss) / 2

    home_pts = possessions * (h_ortg / a_drtg)
    away_pts = possessions * (a_ortg / h_drtg)

    return home_pts + away_pts

def fallback_total(proj):
    return proj * (1 - REGRESSION_WEIGHT) + LEAGUE_AVG_TOTAL * REGRESSION_WEIGHT

def prob_over(proj, line):
    z = (proj - line) / TOTAL_STD_DEV
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

# ============================================================
# UI CONTROLS
# ============================================================
st.title("🏀 College Basketball Predictive Model")

min_prob = st.slider("Minimum Probability (%)", 50, 65, 55)
min_edge = st.slider("Minimum Edge (pts)", 1.0, 6.0, 2.0)

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
        "Line Used": round(line, 1),
        "Line Source": source,
        "Projected Total": round(proj, 1),
        "Edge": round(edge, 1),
        "Prob %": round(prob * 100, 1),
        "Decision": decision
    })

df = pd.DataFrame(rows)
st.dataframe(df.sort_values("Edge", ascending=False), use_container_width=True)
st.caption(f"Games displayed: {len(df)}")
