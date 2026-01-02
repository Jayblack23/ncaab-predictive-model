# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import pandas as pd
import requests
import math
import re
from datetime import date

st.set_page_config(page_title="NCAAB Predictive Betting Model", layout="wide")

# ============================================================
# CONSTANTS
# ============================================================
LEAGUE_AVG_TEMPO = 68
LEAGUE_AVG_EFF = 102
TOTAL_STD_DEV = 11.5
MARKET_CALIBRATION_WEIGHT = 0.25

# ============================================================
# SESSION STATE
# ============================================================
for k in ["bet_log", "clv_log"]:
    if k not in st.session_state:
        st.session_state[k] = []

# ============================================================
# SAFE TEAM NORMALIZATION (FIXED)
# ============================================================
def normalize_team(name):
    if not name:
        return None
    name = name.lower()
    name = re.sub(r"\(.*?\)", "", name)   # remove (FL), (CA)
    name = re.sub(r"[^a-z\s]", "", name)  # remove punctuation
    name = re.sub(r"\s+", " ", name)
    return name.strip()

# ============================================================
# FETCH TEAM METRICS (2025)
# ============================================================
@st.cache_data(ttl=86400)
def fetch_team_metrics():
    headers = {
        "Ocp-Apim-Subscription-Key": st.secrets["SPORTSDATAIO_API_KEY"]
    }
    url = "https://api.sportsdata.io/v3/cbb/stats/json/TeamSeasonStats/2025"
    r = requests.get(url, headers=headers)

    if r.status_code != 200:
        st.error("Failed to load team metrics")
        st.stop()

    rows = []
    for t in r.json():
        rows.append({
            "Team": t["Name"],
            "Key": normalize_team(t["Name"]),
            "Tempo": t.get("PossessionsPerGame", LEAGUE_AVG_TEMPO),
            "AdjOE": t.get("OffensiveEfficiency", LEAGUE_AVG_EFF),
            "AdjDE": t.get("DefensiveEfficiency", LEAGUE_AVG_EFF)
        })

    return pd.DataFrame(rows)

teams = fetch_team_metrics()

# ============================================================
# TEAM LOOKUP WITH FALLBACK MATCHING
# ============================================================
def lookup_team(team_name):
    key = normalize_team(team_name)

    # Exact match
    exact = teams[teams.Key == key]
    if not exact.empty:
        return exact.iloc[0]

    # Fallback: partial match (safe)
    partial = teams[teams.Key.str.contains(key) | key in teams.Key.values]
    if not partial.empty:
        return partial.iloc[0]

    raise ValueError(f"Team not found: {team_name}")

# ============================================================
# FETCH TODAY'S GAMES
# ============================================================
@st.cache_data(ttl=900)
def fetch_todays_games():
    headers = {
        "Ocp-Apim-Subscription-Key": st.secrets["SPORTSDATAIO_API_KEY"]
    }
    today = date.today().strftime("%Y-%m-%d")
    url = f"https://api.sportsdata.io/v3/cbb/scores/json/GamesByDate/{today}"

    r = requests.get(url, headers=headers)
    return r.json() if r.status_code == 200 else []

games = fetch_todays_games()

# ============================================================
# MODEL FUNCTIONS
# ============================================================
def expected_points(tempo, off_eff, def_eff):
    return tempo * (off_eff / def_eff)

def project_total(home, away):
    A = lookup_team(home)
    B = lookup_team(away)

    tempo = (A.Tempo + B.Tempo) / 2
    tempo *= LEAGUE_AVG_TEMPO / tempo

    return (
        expected_points(tempo, A.AdjOE, B.AdjDE) +
        expected_points(tempo, B.AdjOE, A.AdjDE)
    )

def prob_over(projected, line):
    z = (projected - line) / TOTAL_STD_DEV
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

# ============================================================
# UI
# ============================================================
st.title("🏀 College Basketball Predictive Betting Model")

bankroll = st.number_input("Bankroll ($)", value=1000.0)
min_prob = st.slider("Min Probability (%)", 50, 65, 55)
min_edge = st.slider("Min Edge (pts)", 1.0, 5.0, 2.0)

rows = []
games_with_totals = 0
games_processed = 0
games_skipped = 0

# ============================================================
# MAIN LOOP
# ============================================================
for g in games:
    home = g.get("HomeTeam")
    away = g.get("AwayTeam")

    try:
        market_total = float(g.get("OverUnder"))
    except:
        continue

    games_with_totals += 1

    try:
        proj = project_total(home, away)
        edge = round(proj - market_total, 2)
        prob = max(prob_over(proj, market_total), 1 - prob_over(proj, market_total))

        decision = "BET" if prob * 100 >= min_prob and abs(edge) >= min_edge else "PASS"

        rows.append({
            "Game": f"{away} @ {home}",
            "Market Total": market_total,
            "Projected Total": round(proj, 2),
            "Edge": edge,
            "Prob %": round(prob * 100, 1),
            "Decision": decision
        })

        games_processed += 1

    except Exception as e:
        games_skipped += 1

# ============================================================
# DISPLAY
# ============================================================
df = pd.DataFrame(rows)

if df.empty:
    st.error("Games detected but could not be matched. Check team resolution.")
else:
    st.dataframe(df.sort_values("Edge", ascending=False), use_container_width=True)

st.caption(
    f"Totals available: {games_with_totals} | "
    f"Processed: {games_processed} | "
    f"Skipped: {games_skipped}"
)
