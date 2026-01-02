import streamlit as st
import pandas as pd
import requests
import math
from datetime import date

st.set_page_config(page_title="NCAAB Predictive Model", layout="wide")

# ============================================================
# CONFIG
# ============================================================

SPORTSDATAIO_KEY = st.secrets["SPORTSDATAIO_API_KEY"]
ODDS_API_KEY = st.secrets["ODDS_API_KEY"]

SEASON = "2025"
MIN_EFF = 85
MIN_POSSESSIONS = 62

CONF_THRESHOLD = 0.60
EDGE_THRESHOLD = 3.0
BANKROLL = 100.0
UNIT_SIZE = 1.0

# ============================================================
# HELPERS
# ============================================================

def normalize(name: str) -> str:
    return (
        name.lower()
        .replace("state", "st")
        .replace("saint", "st")
        .replace("&", "and")
        .replace(".", "")
        .strip()
    )

# ============================================================
# LOAD TEAM STATS (SPORTSDATAIO)
# ============================================================

@st.cache_data(ttl=86400)
def load_team_stats():
    url = f"https://api.sportsdata.io/v3/cbb/stats/json/TeamSeasonStats/{SEASON}"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATAIO_KEY}
    r = requests.get(url, headers=headers)
    r.raise_for_status()

    teams = {}

    for t in r.json():
        g = t.get("Games", 0)
        if g == 0:
            continue

        fga = t.get("FieldGoalsAttempted", 0)
        fta = t.get("FreeThrowsAttempted", 0)
        orb = t.get("OffensiveRebounds", 0)
        tov = t.get("Turnovers", 0)

        poss = (fga - orb + tov + 0.44 * fta) / g
        poss = max(poss, MIN_POSSESSIONS)

        off_rtg = (t.get("Points", 0) / g) / poss * 100
        def_rtg = (t.get("OpponentPointsPerGame", 0)) / poss * 100

        name = normalize(t["Name"])

        teams[name] = {
            "poss": poss,
            "off": max(off_rtg, MIN_EFF),
            "def": max(def_rtg, MIN_EFF),
        }

    return teams

# ============================================================
# LOAD ODDS (TOTALS)
# ============================================================

@st.cache_data(ttl=300)
def load_odds():
    url = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "totals",
        "oddsFormat": "american",
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()

# ============================================================
# MODEL
# ============================================================

def projected_total(home, away, TEAM):
    h = TEAM[home]
    a = TEAM[away]

    possessions = (h["poss"] + a["poss"]) / 2

    home_pts = possessions * (h["off"] / a["def"])
    away_pts = possessions * (a["off"] / h["def"])

    return round(home_pts + away_pts, 1)

def prob_over(proj, line):
    # Normal approx with historical NCAAB std dev ≈ 11
    return round(1 - (0.5 * (1 + math.erf((line - proj) / (11 * math.sqrt(2))))), 3)

# ============================================================
# SESSION STATE
# ============================================================

if "bet_log" not in st.session_state:
    st.session_state.bet_log = []

# ============================================================
# UI
# ============================================================

st.title("🏀 NCAAB Predictive Totals Model")

TEAM = load_team_stats()
ODDS = load_odds()

rows = []

for g in ODDS:
    home = normalize(g["home_team"])
    away = normalize(g["away_team"])

    if home not in TEAM or away not in TEAM:
        continue

    for b in g["bookmakers"]:
        for m in b["markets"]:
            if m["key"] != "totals":
                continue

            over = next(o for o in m["outcomes"] if o["name"] == "Over")
            line = over["point"]

            proj = projected_total(home, away, TEAM)
            edge = round(proj - line, 2)
            prob = prob_over(proj, line)

            decision = "BET" if prob >= CONF_THRESHOLD and edge >= EDGE_THRESHOLD else "PASS"

            confidence = (
                "⭐⭐⭐" if prob >= 0.65 else
                "⭐⭐" if prob >= 0.60 else
                "⭐"
            )

            stake = round((BANKROLL * 0.01) if decision == "BET" else 0, 2)

            rows.append({
                "Game": f"{g['away_team']} @ {g['home_team']}",
                "Market Total": line,
                "Projected Total": proj,
                "Edge": edge,
                "Probability": round(prob * 100, 1),
                "Confidence": confidence,
                "Decision": decision,
                "Stake": stake
            })

# ============================================================
# DISPLAY
# ============================================================

if not rows:
    st.warning("Games detected, but teams could not be matched yet.")
else:
    df = pd.DataFrame(rows).sort_values("Edge", ascending=False)
    st.dataframe(df, use_container_width=True)

# ============================================================
# ROI / PERFORMANCE
# ============================================================

st.subheader("📈 Performance Summary")

total_bets = len(st.session_state.bet_log)
wins = st.session_state.bet_log.count(1)
losses = st.session_state.bet_log.count(-1)
units = sum(st.session_state.bet_log)

roi = round((units / total_bets) * 100, 2) if total_bets > 0 else 0
win_pct = round((wins / total_bets) * 100, 2) if total_bets > 0 else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Bets", total_bets)
c2.metric("Wins", wins)
c3.metric("Losses", losses)
c4.metric("Units", units)
c5.metric("ROI %", roi)
