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
# SESSION STATE INIT (ROI FIX)
# ============================================================
if "bet_log" not in st.session_state:
    st.session_state.bet_log = []   # +1 win, -1 loss

if "bets" not in st.session_state:
    st.session_state.bets = []

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
# LOAD TEAM METRICS
# ============================================================
@st.cache_data(ttl=86400)
def load_teams():
    headers = {"Ocp-Apim-Subscription-Key": st.secrets["SPORTSDATAIO_API_KEY"]}
    url = "https://api.sportsdata.io/v3/cbb/stats/json/TeamSeasonStats/2025"
    r = requests.get(url, headers=headers)

    rows = []
    for t in r.json():
        rows.append({
            "key": normalize(t["Name"]),
            "name": t["Name"],
            "tempo": t.get("PossessionsPerGame", LEAGUE_AVG_TEMPO),
            "oe": t.get("OffensiveEfficiency", LEAGUE_AVG_EFF),
            "de": t.get("DefensiveEfficiency", LEAGUE_AVG_EFF)
        })

    return pd.DataFrame(rows)

teams_df = load_teams()
TEAM_LOOKUP = {row["key"]: row for _, row in teams_df.iterrows()}

def get_team(name):
    key = normalize(name)
    if key in TEAM_LOOKUP:
        return TEAM_LOOKUP[key]

    for k, v in TEAM_LOOKUP.items():
        if key in k or k in key:
            return v

    raise ValueError(f"Team match failed: {name}")

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
# MODEL FUNCTIONS
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
# UI CONTROLS
# ============================================================
st.title("🏀 College Basketball Predictive Betting Model")

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

if st.button("Project Matchup"):
    proj = projected_total(home, away)
    edge = proj - manual_total
    prob = prob_over(proj, manual_total)

    st.metric("Projected Total", round(proj, 2))
    st.metric("Edge", round(edge, 2))
    st.metric("Over Probability %", round(prob * 100, 1))

# ============================================================
# TODAY'S SLATE
# ============================================================
st.subheader("📅 Today’s Games")

rows = []
skipped = 0

for g in games:
    home = g.get("HomeTeam")
    away = g.get("AwayTeam")
    total = g.get("OverUnder")

    if total in (None, 0):
        skipped += 1
        continue

    try:
        total = float(total)
        if total < 110 or total > 180:
            skipped += 1
            continue

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
            "Decision": decision
        })

        if decision == "BET":
            st.session_state.bets.append(f"{away} @ {home}")

    except Exception as e:
        skipped += 1
        st.write("Skipped:", away, "@", home, "Reason:", e)

df = pd.DataFrame(rows)

if not df.empty:
    st.dataframe(df.sort_values("Edge", ascending=False), use_container_width=True)
else:
    st.error("Games detected but none passed validation.")

st.caption(f"Games processed: {len(df)} | Skipped: {skipped}")

# ============================================================
# ROI TRACKING
# ============================================================
st.subheader("📈 Performance Summary")

if st.session_state.bets:
    st.write("Grade Bets:")

    for i, bet in enumerate(st.session_state.bets):
        col1, col2 = st.columns(2)
        with col1:
            st.write(bet)
        with col2:
            result = st.selectbox(
                "Result",
                ["Pending", "Win", "Loss"],
                key=f"result_{i}"
            )

            if result == "Win":
                st.session_state.bet_log.append(1)
            elif result == "Loss":
                st.session_state.bet_log.append(-1)

total_bets = len(st.session_state.bet_log)
wins = st.session_state.bet_log.count(1)
losses = st.session_state.bet_log.count(-1)
units = sum(st.session_state.bet_log)

roi = round((units / total_bets) * 100, 2) if total_bets else 0
win_pct = round((wins / total_bets) * 100, 2) if total_bets else 0

st.metric("Total Bets", total_bets)
st.metric("Wins", wins)
st.metric("Losses", losses)
st.metric("Units", units)
st.metric("ROI %", roi)
st.metric("Win %", win_pct)
