import streamlit as st
import pandas as pd
import requests
import math

st.set_page_config(page_title="NCAAB Predictive Totals Model", layout="wide")

# ============================================================
# CONFIG
# ============================================================

SPORTSDATAIO_KEY = st.secrets["SPORTSDATAIO_API_KEY"]
ODDS_API_KEY = st.secrets["ODDS_API_KEY"]

SEASON = "2025"

CONF_THRESHOLD = 0.60
EDGE_THRESHOLD = 3.0

MIN_GAME_POSSESSIONS = 68
PACE_SMOOTHING = 0.35
HOME_BONUS = 1.8
LEAGUE_AVG_RTG = 100  # SportsDataIO scale

# ============================================================
# TEAM NAME NORMALIZATION
# ============================================================

ALIASES = {
    "ole miss": "mississippi",
    "uconn": "connecticut",
    "st marys": "saint marys",
    "st johns": "saint johns",
    "unc": "north carolina",
    "uva": "virginia",
}

def normalize(name):
    if not name:
        return ""
    name = (
        name.lower()
        .replace("&", "and")
        .replace(".", "")
        .replace("'", "")
        .strip()
    )
    return ALIASES.get(name, name)

def resolve(name, teams):
    for t in teams:
        if name == t or name in t or t in name:
            return t
    return None

# ============================================================
# LOAD TEAM STATS (SPORTSDATAIO)
# ============================================================

@st.cache_data(ttl=86400)
def load_team_stats():
    url = f"https://api.sportsdata.io/v3/cbb/stats/json/TeamSeasonStats/{SEASON}"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATAIO_KEY}
    data = requests.get(url, headers=headers).json()

    teams = {}

    for t in data:
        g = t.get("Games", 0)
        if g == 0:
            continue

        fga = t.get("FieldGoalsAttempted", 0)
        fta = t.get("FreeThrowsAttempted", 0)
        orb = t.get("OffensiveRebounds", 0)
        tov = t.get("Turnovers", 0)

        poss = (fga - orb + tov + 0.44 * fta) / g

        off_ppg = t.get("PointsPerGame", 0)
        opp_ppg = t.get("OpponentPointsPerGame", 0)

        off_rtg = (off_ppg / poss) * 100 if poss > 0 else 100
        def_rtg = (opp_ppg / poss) * 100 if poss > 0 else 100

        teams[normalize(t["Name"])] = {
            "poss": poss,
            "off": off_rtg,
            "def": def_rtg,
        }

    return teams

# ============================================================
# LOAD ODDS (ODDS API)
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
    return requests.get(url, params=params).json()

# ============================================================
# CORE MODEL (CORRECTED)
# ============================================================

def projected_total(home, away, TEAM):
    h = TEAM[home]
    a = TEAM[away]

    # Smoothed possessions
    raw_poss = (h["poss"] + a["poss"]) / 2
    possessions = raw_poss + PACE_SMOOTHING * (MIN_GAME_POSSESSIONS - raw_poss)

    # Correct efficiency interaction (league-normalized)
    home_ppp = (h["off"] / LEAGUE_AVG_RTG) * (LEAGUE_AVG_RTG / a["def"])
    away_ppp = (a["off"] / LEAGUE_AVG_RTG) * (LEAGUE_AVG_RTG / h["def"])

    total = possessions * (home_ppp + away_ppp)

    # Home-court scoring premium
    total += HOME_BONUS

    return round(total, 1)

def prob_over(proj, line):
    std = 11
    z = (proj - line) / std
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

# ============================================================
# UI
# ============================================================

st.title("🏀 NCAAB Predictive Totals Model")

TEAM = load_team_stats()
ODDS = load_odds()

rows = []

for g in ODDS:
    home_raw = normalize(g.get("home_team"))
    away_raw = normalize(g.get("away_team"))

    home = resolve(home_raw, TEAM)
    away = resolve(away_raw, TEAM)

    if not home or not away:
        continue

    totals = []
    for b in g.get("bookmakers", []):
        for m in b.get("markets", []):
            if m.get("key") == "totals":
                for o in m.get("outcomes", []):
                    if o.get("name") == "Over" and isinstance(o.get("point"), (int, float)):
                        totals.append(o["point"])

    if not totals:
        continue

    market_total = round(sum(totals) / len(totals), 1)
    proj = projected_total(home, away, TEAM)
    edge = round(proj - market_total, 2)
    prob = prob_over(proj, market_total)

    decision = "BET" if prob >= CONF_THRESHOLD and edge >= EDGE_THRESHOLD else "PASS"

    confidence = (
        "⭐⭐⭐" if prob >= 0.65 else
        "⭐⭐" if prob >= 0.60 else
        "⭐"
    )

    rows.append({
        "Game": f"{g['away_team']} @ {g['home_team']}",
        "Market Total": market_total,
        "Projected Total": proj,
        "Edge": edge,
        "Probability %": round(prob * 100, 1),
        "Confidence": confidence,
        "Decision": decision,
    })

# ============================================================
# DISPLAY
# ============================================================

if not rows:
    st.warning("Games detected, but no valid totals available yet.")
else:
    df = pd.DataFrame(rows).sort_values("Edge", ascending=False)
    st.dataframe(df, use_container_width=True)
