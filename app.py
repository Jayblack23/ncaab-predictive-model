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

MIN_POSSESSIONS = 62
MIN_EFF = 85

CONF_THRESHOLD = 0.60
EDGE_THRESHOLD = 3.0
BANKROLL = 100.0

# ============================================================
# TEAM NAME NORMALIZATION + ALIASES
# ============================================================

TEAM_ALIASES = {
    "ole miss": "mississippi",
    "uconn": "connecticut",
    "st marys": "saint marys",
    "st johns": "saint johns",
    "unc": "north carolina",
    "uva": "virginia",
}

def normalize(name: str) -> str:
    if not name:
        return ""
    n = (
        name.lower()
        .replace("&", "and")
        .replace(".", "")
        .replace("'", "")
        .replace("state", "st")
        .replace("saint", "saint")
        .strip()
    )
    return TEAM_ALIASES.get(n, n)

def resolve_team(name, TEAM):
    if name in TEAM:
        return name
    for t in TEAM:
        if name in t or t in name:
            return t
    return None

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

        teams[normalize(t["Name"])] = {
            "poss": poss,
            "off": max(off_rtg, MIN_EFF),
            "def": max(def_rtg, MIN_EFF),
        }

    return teams

# ============================================================
# LOAD ODDS (CONSENSUS TOTALS)
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

    # CORRECT KenPom-style formula
    home_pts = possessions * ((h["off"] + a["def"]) / 2) / 100
    away_pts = possessions * ((a["off"] + h["def"]) / 2) / 100

    return round(home_pts + away_pts, 1)

def prob_over(proj, line):
    std = 11  # historical NCAAB total std dev
    z = (proj - line) / std
    return round(0.5 * (1 + math.erf(z / math.sqrt(2))), 3)

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
    home_raw = normalize(g.get("home_team", ""))
    away_raw = normalize(g.get("away_team", ""))

    home = resolve_team(home_raw, TEAM)
    away = resolve_team(away_raw, TEAM)

    if not home or not away:
        continue

    # ---- CONSENSUS MARKET TOTAL ----
    lines = []
    for b in g.get("bookmakers", []):
        for m in b.get("markets", []):
            if m.get("key") == "totals":
                for o in m.get("outcomes", []):
                    if o.get("name") == "Over" and isinstance(o.get("point"), (int, float)):
                        lines.append(o["point"])

    if not lines:
        continue

    market_total = round(sum(lines) / len(lines), 1)

    proj = projected_total(home, away, TEAM)
    edge = round(proj - market_total, 2)
    prob = prob_over(proj, market_total)

    decision = "BET" if prob >= CONF_THRESHOLD and edge >= EDGE_THRESHOLD else "PASS"

    confidence = (
        "⭐⭐⭐" if prob >= 0.65 else
        "⭐⭐" if prob >= 0.60 else
        "⭐"
    )

    stake = round(BANKROLL * 0.01, 2) if decision == "BET" else 0

    rows.append({
        "Game": f"{g['away_team']} @ {g['home_team']}",
        "Market Total": market_total,
        "Projected Total": proj,
        "Edge": edge,
        "Probability %": round(prob * 100, 1),
        "Confidence": confidence,
        "Decision": decision,
        "Stake": stake
    })

# ============================================================
# DISPLAY
# ============================================================

if not rows:
    st.warning("Games detected, but none could be matched to team metrics yet.")
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
