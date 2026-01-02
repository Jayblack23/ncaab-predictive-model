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
# SESSION STATE
# ============================================================
if "bet_log" not in st.session_state:
    st.session_state.bet_log = []  # +1 / -1 results

if "bankroll" not in st.session_state:
    st.session_state.bankroll = 100.0

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

KELLY_FRACTION = 0.5
MAX_STAKE_PCT = 5.0

AUTO_BET_PROB = 0.60
AUTO_BET_EDGE = 3.0

# ============================================================
# LOAD TEAM STATS (SPORTSDATAIO)
# ============================================================
@st.cache_data(ttl=86400)
def load_teams():
    headers = {"Ocp-Apim-Subscription-Key": st.secrets["SPORTSDATAIO_API_KEY"]}
    url = "https://api.sportsdata.io/v3/cbb/stats/json/TeamSeasonStats/2025"
    r = requests.get(url, headers=headers)

    rows = []
    for t in r.json():
        g = t.get("Games", 0)
        if g == 0:
            continue

        rows.append({
            "TeamID": t["TeamID"],
            "Name": t["Name"],
            "Games": g,
            "Points": t.get("Points", 0),
            "OppPoints": t.get("OpponentPointsPerGame", 0) * g,
            "FGA": t.get("FieldGoalsAttempted", 0),
            "FTA": t.get("FreeThrowsAttempted", 0),
            "ORB": t.get("OffensiveRebounds", 0),
            "TOV": t.get("Turnovers", 0),
        })

    return pd.DataFrame(rows)

teams_df = load_teams()
TEAM = teams_df.set_index("TeamID").to_dict("index")
NAME = teams_df.set_index("TeamID")["Name"].to_dict()

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
def possessions(t):
    p = (t["FGA"] - t["ORB"] + t["TOV"] + 0.44 * t["FTA"]) / t["Games"]
    return max(p, MIN_POSSESSIONS)

def ratings(t):
    poss = possessions(t)
    ortg = (t["Points"] / t["Games"]) / poss * 100
    drtg = (t["OppPoints"] / t["Games"]) / poss * 100
    return poss, max(ortg, MIN_EFF), max(drtg, MIN_EFF)

def projected_total(home_id, away_id):
    hp, ho, hd = ratings(TEAM[home_id])
    ap, ao, ad = ratings(TEAM[away_id])
    poss = (hp + ap) / 2 * HOME_PACE_BONUS
    return poss * (ho / ad + ao / hd)

def fallback_total(p):
    reg = p * (1 - REGRESSION_WEIGHT) + LEAGUE_AVG_TOTAL * REGRESSION_WEIGHT
    return min(max(reg, MIN_FALLBACK_TOTAL), MAX_FALLBACK_TOTAL)

def prob_over(p, l):
    z = (p - l) / TOTAL_STD_DEV
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

def kelly_stake(prob):
    raw = (prob - (1 - prob)) * KELLY_FRACTION
    return min(max(raw * 100, 0), MAX_STAKE_PCT)

# ============================================================
# UI CONTROLS
# ============================================================
st.title("🏀 College Basketball Totals Betting Model")

st.session_state.bankroll = st.number_input(
    "Current Bankroll ($)",
    min_value=10.0,
    value=st.session_state.bankroll,
    step=10.0
)

min_prob = st.slider("Minimum Probability (%)", 50, 65, 55)
min_edge = st.slider("Minimum Edge (points)", 1.0, 6.0, 2.0)

# ============================================================
# BUILD TODAY'S SLATE
# ============================================================
rows = []

for g in games:
    h, a = g.get("HomeTeamID"), g.get("AwayTeamID")
    if h not in TEAM or a not in TEAM:
        continue

    proj = projected_total(h, a)
    market = g.get("OverUnder")

    if market and VALID_TOTAL_RANGE[0] <= market <= VALID_TOTAL_RANGE[1]:
        line = float(market)
        source = "Market"
        adj_prob, adj_edge = min_prob, min_edge
    else:
        line = fallback_total(proj)
        source = "Fallback"
        adj_prob, adj_edge = min_prob + 3, min_edge + 1

    p_over = prob_over(proj, line)
    p_under = 1 - p_over

    if p_over >= p_under:
        side, prob, edge = "OVER", p_over, proj - line
    else:
        side, prob, edge = "UNDER", p_under, line - proj

    # AUTO BET RULE
    auto_bet = prob >= AUTO_BET_PROB and abs(edge) >= AUTO_BET_EDGE

    decision = (
        "BET"
        if auto_bet or (prob * 100 >= adj_prob and abs(edge) >= adj_edge)
        else "PASS"
    )

    # CONFIDENCE STARS
    if prob >= 0.62:
        confidence = "⭐⭐⭐"
    elif prob >= 0.58:
        confidence = "⭐⭐"
    else:
        confidence = "⭐"

    stake_pct = kelly_stake(prob)
    stake_amt = round(st.session_state.bankroll * stake_pct / 100, 2)

    rows.append({
        "Game": f"{NAME[a]} @ {NAME[h]}",
        "Side": side,
        "Line": round(line, 1),
        "Projected Total": round(proj, 1),
        "Edge": round(edge, 1),
        "Prob %": round(prob * 100, 1),
        "Confidence": confidence,
        "Stake $": stake_amt,
        "Decision": decision
    })

df = pd.DataFrame(rows).sort_values("Edge", ascending=False)
st.dataframe(df, use_container_width=True)

# ============================================================
# ROI / PERFORMANCE SUMMARY
# ============================================================
st.subheader("📈 Performance Summary")

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
