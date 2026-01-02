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

if "clv_log" not in st.session_state:
    st.session_state.clv_log = []

if "graded_games" not in st.session_state:
    st.session_state.graded_games = set()

# ============================================================
# CONSTANTS / GUARDRAILS
# ============================================================
TOTAL_STD_DEV = 11.5
REGRESSION_WEIGHT = 0.15

LEAGUE_AVG_TOTAL = 142
MIN_FALLBACK_TOTAL = 125
MAX_FALLBACK_TOTAL = 165
VALID_TOTAL_RANGE = (120, 170)

MIN_POSSESSIONS = 60
MIN_EFF = 80

HOME_PACE_BONUS = 1.02
AWAY_PACE_PENALTY = 0.98
RECENT_WEIGHT = 0.65

# ============================================================
# LOAD TEAM STATS (SPORTSDATAIO – SAFE SCHEMA)
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
# MODEL CORE
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

    return poss, max(ortg, MIN_EFF), max(drtg, MIN_EFF)

def projected_total(home_id, away_id):
    home = TEAM[home_id]
    away = TEAM[away_id]

    h_poss, h_ortg, h_drtg = team_ratings(home)
    a_poss, a_ortg, a_drtg = team_ratings(away)

    # Opponent-adjusted tempo
    possessions = (h_poss + a_poss) / 2

    # Home/Away pace adjustment
    possessions *= HOME_PACE_BONUS

    home_pts = possessions * (h_ortg / a_drtg)
    away_pts = possessions * (a_ortg / h_drtg)

    return home_pts + away_pts

def fallback_total(proj):
    regressed = proj * (1 - REGRESSION_WEIGHT) + LEAGUE_AVG_TOTAL * REGRESSION_WEIGHT
    return min(max(regressed, MIN_FALLBACK_TOTAL), MAX_FALLBACK_TOTAL)

def prob_over(proj, line):
    z = (proj - line) / TOTAL_STD_DEV
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

# ============================================================
# UI CONTROLS
# ============================================================
st.title("🏀 College Basketball Predictive Betting Model")

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

    if market is not None and VALID_TOTAL_RANGE[0] <= market <= VALID_TOTAL_RANGE[1]:
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

# ============================================================
# ROI + CLV TRACKING
# ============================================================
st.subheader("📈 Performance & CLV Tracking")

for i, row in df[df["Decision"] == "BET"].iterrows():
    game = row["Game"]
    open_line = row["Line Used"]

    if game in st.session_state.graded_games:
        continue

    close_line = st.number_input(
        f"{game} – Closing Line",
        min_value=120.0,
        max_value=170.0,
        step=0.5,
        key=f"cl_{i}"
    )

    result = st.selectbox(
        f"{game} – Result",
        ["Pending", "Win", "Loss"],
        key=f"res_{i}"
    )

    if result in ["Win", "Loss"] and close_line > 0:
        st.session_state.graded_games.add(game)
        st.session_state.bet_log.append(1 if result == "Win" else -1)
        st.session_state.clv_log.append(open_line - close_line)

# ============================================================
# METRICS
# ============================================================
total = len(st.session_state.bet_log)
wins = st.session_state.bet_log.count(1)
losses = st.session_state.bet_log.count(-1)
units = sum(st.session_state.bet_log)
avg_clv = round(sum(st.session_state.clv_log) / len(st.session_state.clv_log), 2) if st.session_state.clv_log else 0

st.metric("Total Bets", total)
st.metric("Wins", wins)
st.metric("Losses", losses)
st.metric("Units", units)
st.metric("ROI %", round((units / total) * 100, 2) if total else 0)
st.metric("Avg CLV (pts)", avg_clv)
