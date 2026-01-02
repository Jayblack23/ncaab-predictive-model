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

# ============================================================
# TEAM NAME NORMALIZATION
# ============================================================
def normalize(name):
    if not name:
        return None
    name = name.lower()
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"[^a-z\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()

# ============================================================
# LOAD TEAM METRICS (2025)
# ============================================================
@st.cache_data(ttl=86400)
def load_teams():
    headers = {"Ocp-Apim-Subscription-Key": st.secrets["SPORTSDATAIO_API_KEY"]}
    url = "https://api.sportsdata.io/v3/cbb/stats/json/TeamSeasonStats/2025"
    r = requests.get(url, headers=headers)
    data = r.json()

    rows = []
    for t in data:
        rows.append({
            "key": normalize(t["Name"]),
            "name": t["Name"],
            "tempo": t.get("PossessionsPerGame", LEAGUE_AVG_TEMPO),
            "oe": t.get("OffensiveEfficiency", LEAGUE_AVG_EFF),
            "de": t.get("DefensiveEfficiency", LEAGUE_AVG_EFF),
        })

    return pd.DataFrame(rows)

teams_df = load_teams()
TEAM_LOOKUP = {row["key"]: row for _, row in teams_df.iterrows()}

def get_team(name):
    key = normalize(name)
    if key in TEAM_LOOKUP:
        return TEAM_LOOKUP[key]
    for k in TEAM_LOOKUP:
        if key in k or k in key:
            return TEAM_LOOKUP[k]
    raise ValueError(f"Team not found: {name}")

# ============================================================
# FETCH TODAY'S GAMES
# ============================================================
@st.cache_data(ttl=900)
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
def expected_points(tempo, oe, de):
    return tempo * (oe / de)

def projected_total(home, away):
    A = get_team(home)
    B = get_team(away)

    tempo = (A["tempo"] + B["tempo"]) / 2
    tempo *= LEAGUE_AVG_TEMPO / tempo

    return (
        expected_points(tempo, A["oe"], B["de"]) +
        expected_points(tempo, B["oe"], A["de"])
    )

def prob_over(proj, line):
    z = (proj - line) / TOTAL_STD_DEV
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

# ============================================================
# CONFIDENCE GRADING
# ============================================================
def confidence_grade(edge, prob):
    if abs(edge) >= 5 and prob >= 0.62:
        return "A"
    elif abs(edge) >= 3 and prob >= 0.58:
        return "B"
    else:
        return "C"

# ============================================================
# SHARP VS PUBLIC INDICATOR (PROXY)
# ============================================================
def sharp_indicator(edge):
    if abs(edge) >= 4:
        return "Sharp Side"
    elif abs(edge) >= 2:
        return "Lean"
    else:
        return "Public / No Edge"

# ============================================================
# UI CONTROLS
# ============================================================
st.title("🏀 College Basketball Predictive Betting Model")

bankroll = st.number_input("Bankroll ($)", 1000.0)
min_prob = st.slider("Min Probability (%)", 50, 65, 55)
min_edge = st.slider("Min Edge (pts)", 1.0, 6.0, 2.0)

# ============================================================
# MANUAL MATCHUP TOOL
# ============================================================
st.subheader("🧪 Manual Matchup")

teams = sorted(teams_df["name"].unique())
c1, c2, c3 = st.columns(3)

with c1:
    away = st.selectbox("Away Team", teams)
with c2:
    home = st.selectbox("Home Team", teams)
with c3:
    manual_total = st.number_input("Market Total", 110.0, 180.0, 145.5)

if st.button("Run Projection"):
    proj = projected_total(home, away)
    edge = proj - manual_total
    prob = prob_over(proj, manual_total)

    st.metric("Projected Total", round(proj, 2))
    st.metric("Edge", round(edge, 2))
    st.metric("Over Probability %", round(prob * 100, 1))
    st.metric("Confidence", confidence_grade(edge, prob))

# ============================================================
# TODAY'S SLATE
# ============================================================
st.subheader("📅 Today’s Games")

rows = []

for g in games:
    home = g.get("HomeTeam")
    away = g.get("AwayTeam")

    try:
        total = float(g.get("OverUnder"))
    except:
        continue

    # sanity filter
    if total < 110 or total > 180:
        continue

    try:
        proj = projected_total(home, away)
        edge = proj - total
        p_over = prob_over(proj, total)
        prob = max(p_over, 1 - p_over)

        decision = "BET" if prob * 100 >= min_prob and abs(edge) >= min_edge else "PASS"

        rows.append({
            "Game": f"{away} @ {home}",
            "Market Total": total,
            "Projected Total": round(proj, 2),
            "Edge": round(edge, 2),
            "Prob %": round(prob * 100, 1),
            "Confidence": confidence_grade(edge, prob),
            "Sharp/Public": sharp_indicator(edge),
            "Decision": decision
        })

    except:
        continue

df = pd.DataFrame(rows)

if df.empty:
    st.warning("Games detected, but totals or team matching incomplete.")
else:
    st.dataframe(df.sort_values("Edge", ascending=False), use_container_width=True)

st.caption(f"Games with usable totals: {len(df)}")
