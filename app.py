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

# Kelly safety
KELLY_FRACTION = 0.5
MAX_STAKE_PCT = 5.0

# ============================================================
# LOAD TEAM STATS
# ============================================================
@st.cache_data(ttl=86400)
def load_teams():
    headers = {"Ocp-Apim-Subscription-Key": st.secrets["SPORTSDATAIO_API_KEY"]}
    url = "https://api.sportsdata.io/v3/cbb/stats/json/TeamSeasonStats/2025"
    r = requests.get(url, headers=headers)

    teams = []
    for t in r.json():
        g = t.get("Games", 0)
        if g == 0:
            continue

        teams.append({
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

    return pd.DataFrame(teams)

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

def projected_total(h, a):
    hp, ho, hd = ratings(TEAM[h])
    ap, ao, ad = ratings(TEAM[a])

    poss = (hp + ap) / 2 * HOME_PACE_BONUS
    home_pts = poss * (ho / ad)
    away_pts = poss * (ao / hd)
    return home_pts + away_pts

def fallback_total(p):
    reg = p * (1 - REGRESSION_WEIGHT) + LEAGUE_AVG_TOTAL * REGRESSION_WEIGHT
    return min(max(reg, MIN_FALLBACK_TOTAL), MAX_FALLBACK_TOTAL)

def prob_over(p, l):
    z = (p - l) / TOTAL_STD_DEV
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

def kelly(prob, edge):
    raw = ((prob * (edge / TOTAL_STD_DEV)) - (1 - prob)) * KELLY_FRACTION
    return min(max(raw * 100, 0), MAX_STAKE_PCT)

# ============================================================
# UI
# ============================================================
st.title("🏀 College Basketball Totals Model")

min_prob = st.slider("Min Probability (%)", 50, 65, 55)
min_edge = st.slider("Min Edge (pts)", 1.0, 6.0, 2.0)

# ============================================================
# SLATE
# ============================================================
rows = []

for g in games:
    h = g.get("HomeTeamID")
    a = g.get("AwayTeamID")
    if h not in TEAM or a not in TEAM:
        continue

    proj = projected_total(h, a)
    market = g.get("OverUnder")

    if market and VALID_TOTAL_RANGE[0] <= market <= VALID_TOTAL_RANGE[1]:
        line = float(market)
        source = "Market"
        adj_prob = min_prob
        adj_edge = min_edge
    else:
        line = fallback_total(proj)
        source = "Fallback"
        adj_prob = min_prob + 3
        adj_edge = min_edge + 1

    p_over = prob_over(proj, line)
    p_under = 1 - p_over

    edge_over = proj - line
    edge_under = -edge_over

    if p_over >= p_under:
        side = "OVER"
        prob = p_over
        edge = edge_over
    else:
        side = "UNDER"
        prob = p_under
        edge = edge_under

    decision = "BET" if prob * 100 >= adj_prob and abs(edge) >= adj_edge else "PASS"

    stars = "⭐⭐⭐" if prob >= 0.6 else "⭐⭐" if prob >= 0.57 else "⭐"
    stake = kelly(prob, abs(edge))

    rows.append({
        "Game": f"{NAME[a]} @ {NAME[h]}",
        "Side": side,
        "Line": round(line, 1),
        "Source": source,
        "Proj Total": round(proj, 1),
        "Edge": round(edge, 1),
        "Prob %": round(prob * 100, 1),
        "Confidence": stars,
        "Stake %": round(stake, 2),
        "Decision": decision
    })

df = pd.DataFrame(rows).sort_values("Edge", ascending=False)
st.dataframe(df, use_container_width=True)
