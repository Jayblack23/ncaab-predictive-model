# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import pandas as pd
import requests
import math
from datetime import date

st.set_page_config(page_title="NCAAB Totals Betting Model", layout="wide")

# ============================================================
# SESSION STATE
# ============================================================
if "bankroll" not in st.session_state:
    st.session_state.bankroll = 100.0

if "bet_log" not in st.session_state:
    st.session_state.bet_log = []

# ============================================================
# CONSTANTS
# ============================================================
TOTAL_STD = 11.0
HOME_PACE_MULT = 1.01
LEAGUE_AVG_TOTAL = 142

MIN_POSSESSIONS = 62
MIN_EFF = 85

AUTO_BET_PROB = 0.60
AUTO_BET_EDGE = 3.0

KELLY_FRACTION = 0.5
MAX_STAKE_PCT = 5.0

# ============================================================
# LOAD TEAM STATS (SPORTSDATAIO — SAFE)
# ============================================================
@st.cache_data(ttl=86400)
def load_team_stats():
    headers = {"Ocp-Apim-Subscription-Key": st.secrets["SPORTSDATAIO_API_KEY"]}
    url = "https://api.sportsdata.io/v3/cbb/stats/json/TeamSeasonStats/2025"
    r = requests.get(url, headers=headers)

    teams = {}
    names = {}

    for t in r.json():
        g = t.get("Games")
        pts = t.get("Points")
        opp_ppg = t.get("OpponentPointsPerGame")

        if not g or not pts or not opp_ppg:
            continue

        fga = t.get("FieldGoalsAttempted", 0)
        fta = t.get("FreeThrowsAttempted", 0)
        orb = t.get("OffensiveRebounds", 0)
        tov = t.get("Turnovers", 0)

        poss = (fga - orb + tov + 0.44 * fta) / g
        poss = max(poss, MIN_POSSESSIONS)

        off_rtg = (pts / g) / poss * 100
        def_rtg = opp_ppg / poss * 100

        teams[t["TeamID"]] = {
            "poss": poss,
            "off": max(off_rtg, MIN_EFF),
            "def": max(def_rtg, MIN_EFF),
        }

        names[t["TeamID"]] = t["Name"]

    return teams, names

TEAM, NAME = load_team_stats()

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
# LOAD ODDS (ODDS API)
# ============================================================
@st.cache_data(ttl=300)
def load_odds():
    url = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds"
    params = {
        "apiKey": st.secrets["ODDS_API_KEY"],
        "regions": "us",
        "markets": "totals",
        "oddsFormat": "decimal"
    }
    r = requests.get(url, params=params)
    return r.json() if r.status_code == 200 else []

odds_raw = load_odds()

ODDS = {}
for g in odds_raw:
    home = g.get("home_team", "").lower()
    away = g.get("away_team", "").lower()

    for b in g.get("bookmakers", []):
        for m in b.get("markets", []):
            if m.get("key") == "totals":
                for o in m.get("outcomes", []):
                    if "point" in o:
                        ODDS[(away, home)] = float(o["point"])
                        break

# ============================================================
# MODEL FUNCTIONS
# ============================================================
def projected_total(home_id, away_id):
    h = TEAM[home_id]
    a = TEAM[away_id]

    poss = (h["poss"] + a["poss"]) / 2 * HOME_PACE_MULT

    home_ppp = (h["off"] + a["def"]) / 200
    away_ppp = (a["off"] + h["def"]) / 200

    return poss * (home_ppp + away_ppp)

def prob_over(projected, line):
    z = (projected - line) / TOTAL_STD
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

def kelly(prob):
    raw = (prob - (1 - prob)) * KELLY_FRACTION
    return min(max(raw * 100, 0), MAX_STAKE_PCT)

# ============================================================
# UI
# ============================================================
st.title("🏀 College Basketball Totals Betting Model")

st.session_state.bankroll = st.number_input(
    "Bankroll ($)", min_value=10.0, value=st.session_state.bankroll, step=10.0
)

min_prob = st.slider("Minimum Probability %", 50, 65, 55)
min_edge = st.slider("Minimum Edge (pts)", 1.0, 6.0, 2.0)

# ============================================================
# BUILD SLATE
# ============================================================
rows = []

for g in games:
    h = g.get("HomeTeamID")
    a = g.get("AwayTeamID")

    if h not in TEAM or a not in TEAM:
        continue

    proj = projected_total(h, a)

    key = (NAME[a].lower(), NAME[h].lower())
    if key in ODDS:
        line = ODDS[key]
        source = "Market"
    else:
        line = LEAGUE_AVG_TOTAL
        source = "Implied"

    p_over = prob_over(proj, line)
    p_under = 1 - p_over

    if p_over >= p_under:
        side = "OVER"
        prob = p_over
        edge = proj - line
    else:
        side = "UNDER"
        prob = p_under
        edge = line - proj

    decision = "BET" if (prob >= AUTO_BET_PROB and edge >= AUTO_BET_EDGE) else "PASS"

    confidence = "⭐⭐⭐" if prob >= 0.62 else "⭐⭐" if prob >= 0.58 else "⭐"
    stake_pct = kelly(prob)
    stake_amt = round(st.session_state.bankroll * stake_pct / 100, 2)

    rows.append({
        "Game": f"{NAME[a]} @ {NAME[h]}",
        "Side": side,
        "Line": round(line, 1),
        "Line Source": source,
        "Projected Total": round(proj, 1),
        "Edge": round(edge, 1),
        "Prob %": round(prob * 100, 1),
        "Confidence": confidence,
        "Stake $": stake_amt,
        "Decision": decision
    })

# ============================================================
# DISPLAY RESULTS (SAFE)
# ============================================================
if len(rows) == 0:
    st.warning("No valid games available yet. Betting lines may not be posted.")
else:
    df = pd.DataFrame(rows)

    if "Edge" in df.columns:
        df = df.sort_values("Edge", ascending=False)

    st.dataframe(df, use_container_width=True)

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

st.metric("Total Bets", total_bets)
st.metric("Wins", wins)
st.metric("Losses", losses)
st.metric("Units", units)
st.metric("ROI %", roi)
st.metric("Win %", win_pct)
