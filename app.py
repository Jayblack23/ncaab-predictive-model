# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import pandas as pd
import requests
import math
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
# FETCH TEAM METRICS (2025 SEASON)
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
            "Tempo": t.get("PossessionsPerGame", LEAGUE_AVG_TEMPO),
            "AdjOE": t.get("OffensiveEfficiency", LEAGUE_AVG_EFF),
            "AdjDE": t.get("DefensiveEfficiency", LEAGUE_AVG_EFF)
        })

    return pd.DataFrame(rows)

teams = fetch_team_metrics()

# ============================================================
# FETCH TODAY’S GAMES
# ============================================================
@st.cache_data(ttl=900)
def fetch_todays_games():
    headers = {
        "Ocp-Apim-Subscription-Key": st.secrets["SPORTSDATAIO_API_KEY"]
    }
    today = date.today().strftime("%Y-%m-%d")
    url = f"https://api.sportsdata.io/v3/cbb/scores/json/GamesByDate/{today}"

    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return []

    return r.json()

games = fetch_todays_games()

# ============================================================
# MODEL FUNCTIONS (SAFE)
# ============================================================
def expected_points(tempo, off_eff, def_eff):
    return tempo * (off_eff / def_eff)

def project_total(home, away):
    A_df = teams.loc[teams.Team == home]
    B_df = teams.loc[teams.Team == away]

    if A_df.empty or B_df.empty:
        raise ValueError("Team not found in metrics")

    A = A_df.iloc[0]
    B = B_df.iloc[0]

    tempo = (A.Tempo + B.Tempo) / 2
    tempo *= LEAGUE_AVG_TEMPO / tempo

    return (
        expected_points(tempo, A.AdjOE, B.AdjDE) +
        expected_points(tempo, B.AdjOE, A.AdjDE)
    )

def prob_over(projected, line):
    z = (projected - line) / TOTAL_STD_DEV
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

def kelly_fraction(prob, odds=-110):
    b = abs(odds) / 100
    return max((prob * (b + 1) - 1) / b, 0)

# ============================================================
# UI CONTROLS
# ============================================================
st.title("🏀 College Basketball Predictive Betting Model")

bankroll = st.number_input("Bankroll ($)", value=1000.0, step=100.0)
min_prob = st.slider("Minimum Probability (%)", 50, 65, 55)
min_edge = st.slider("Minimum Edge (pts)", 1.0, 5.0, 2.0)

st.subheader("📅 Today’s Games")

rows = []
games_with_totals = 0
games_processed = 0
games_skipped = 0

# ============================================================
# MAIN GAME LOOP (FULLY SAFE)
# ============================================================
for g in games:
    home = g.get("HomeTeam")
    away = g.get("AwayTeam")
    market_total_raw = g.get("OverUnder")

    if not home or not away:
        continue

    # Convert market total safely
    try:
        market_total = float(market_total_raw)
    except (TypeError, ValueError):
        continue

    games_with_totals += 1

    try:
        raw_proj = project_total(home, away)

        proj = (
            raw_proj * (1 - MARKET_CALIBRATION_WEIGHT) +
            market_total * MARKET_CALIBRATION_WEIGHT
        )

        edge = round(proj - market_total, 2)
        p_over = prob_over(proj, market_total)
        p_under = 1 - p_over

        side = "OVER" if edge > 0 else "UNDER"
        prob = max(p_over, p_under)

        decision = (
            "BET"
            if prob * 100 >= min_prob and abs(edge) >= min_edge
            else "PASS"
        )

        stake = round(bankroll * kelly_fraction(prob), 2)

        rows.append({
            "Game": f"{away} @ {home}",
            "Market Total": market_total,
            "Projected Total": round(proj, 2),
            "Edge": edge,
            "Side": side,
            "Prob %": round(prob * 100, 1),
            "Kelly $": stake,
            "Decision": decision
        })

        games_processed += 1

    except ValueError:
        games_skipped += 1

# ============================================================
# DISPLAY RESULTS
# ============================================================
df = pd.DataFrame(rows)

if df.empty and games_with_totals > 0:
    st.error("Totals exist, but teams could not be matched to metrics.")
elif df.empty:
    st.warning("No games with posted totals yet.")
else:
    df = df.sort_values("Edge", ascending=False)
    st.dataframe(df, use_container_width=True)

st.caption(
    f"Games with totals: {games_with_totals} | "
    f"Processed: {games_processed} | "
    f"Skipped (team mismatch): {games_skipped}"
)

# ============================================================
# BET LOGGING
# ============================================================
st.subheader("🧾 Log Bet Result")

c1, c2, c3 = st.columns(3)
with c1:
    result = st.selectbox("Result", ["Win", "Loss"])
with c2:
    open_line = st.number_input("Open Line", step=0.5)
with c3:
    close_line = st.number_input("Close Line", step=0.5)

if st.button("Save Bet"):
    st.session_state.bet_log.append(1 if result == "Win" else -1)
    st.session_state.clv_log.append(open_line - close_line)

# ============================================================
# PERFORMANCE SUMMARY
# ============================================================
st.subheader("📈 Performance Summary")

bets = len(st.session_state.bet_log)
units = sum(st.session_state.bet_log)
roi = round((units / bets) * 100, 2) if bets else 0
avg_clv = round(sum(st.session_state.clv_log) / len(st.session_state.clv_log), 2) if st.session_state.clv_log else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Bets", bets)
m2.metric("Units", units)
m3.metric("ROI %", roi)
m4.metric("Avg CLV", avg_clv)

st.caption("Data: SportsDataIO · For informational purposes only")
