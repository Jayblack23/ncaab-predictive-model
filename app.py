# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import pandas as pd
import requests
import os
from math import erf, sqrt
from datetime import datetime
import numpy as np

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NCAAB Totals Model",
    layout="wide"
)

# ============================================================
# CONSTANTS
# ============================================================
BANKROLL = 1000
ODDS = -110
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

BET_LOG_FILE = f"{DATA_DIR}/bet_log.csv"
LINE_HISTORY_FILE = f"{DATA_DIR}/line_history.csv"

# ============================================================
# INITIALIZE STORAGE
# ============================================================
if not os.path.exists(BET_LOG_FILE):
    pd.DataFrame(columns=[
        "Game","Side","Bet_Line",
        "Prob","Edge","Units","Result","Date"
    ]).to_csv(BET_LOG_FILE, index=False)

if not os.path.exists(LINE_HISTORY_FILE):
    pd.DataFrame(columns=["Game","Line","Time"]).to_csv(
        LINE_HISTORY_FILE, index=False
    )

bet_log = pd.read_csv(BET_LOG_FILE)
line_history = pd.read_csv(LINE_HISTORY_FILE)

# ============================================================
# MATH FUNCTIONS
# ============================================================
def normal_cdf(x):
    return (1 + erf(x / sqrt(2))) / 2

def implied_prob(odds):
    return abs(odds) / (abs(odds) + 100)

def prob_over(line, proj, std):
    return 1 - normal_cdf((line - proj) / std)

def prob_under(line, proj, std):
    return normal_cdf((line - proj) / std)

def kelly_fraction(p, odds):
    b = 100 / abs(odds)
    return max((p*b - (1-p)) / b, 0)

# ============================================================
# FETCH ODDS
# ============================================================
@st.cache_data(ttl=300)
def fetch_totals():
    r = requests.get(
        "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds",
        params={
            "apiKey": st.secrets["ODDS_API_KEY"],
            "regions": "us",
            "markets": "totals",
            "oddsFormat": "american"
        }
    )

    data = r.json()
    rows = []

    if not isinstance(data, list):
        return pd.DataFrame()

    for g in data:
        try:
            totals = [
                o["point"]
                for b in g["bookmakers"]
                for o in b["markets"][0]["outcomes"]
            ]

            rows.append({
                "Game": f"{g['away_team']} vs {g['home_team']}",
                "Market_Total": np.mean(totals)
            })
        except:
            continue

    return pd.DataFrame(rows)

# ============================================================
# UI HEADER
# ============================================================
st.title("🏀 NCAAB Totals Predictive Model")
st.caption("Calibrated • Directional • Market-Validated")

if st.button("🔄 Refresh Odds"):
    st.experimental_rerun()

df = fetch_totals()
if df.empty:
    st.error("No odds available.")
    st.stop()

# ============================================================
# TEAM METRICS UPLOAD (FEATURE #1)
# ============================================================
st.subheader("📂 Upload Team Metrics CSV")
st.caption("Required columns: Team, Tempo, AdjOE")

team_file = st.file_uploader("Upload CSV", type="csv")
if team_file is None:
    st.stop()

teams = pd.read_csv(team_file)

def team_stat(game, col):
    away, home = game.split(" vs ")
    try:
        return (
            teams.loc[teams.Team == home, col].values[0],
            teams.loc[teams.Team == away, col].values[0]
        )
    except:
        return None, None

df["Tempo_H"], df["Tempo_A"] = zip(*df.Game.map(lambda g: team_stat(g,"Tempo")))
df["OE_H"], df["OE_A"] = zip(*df.Game.map(lambda g: team_stat(g,"AdjOE")))
df.dropna(inplace=True)

# ============================================================
# PROJECTION ENGINE
# ============================================================
df["Tempo"] = (df.Tempo_H + df.Tempo_A) / 2
df["Raw_Model"] = df.Tempo * (df.OE_H + df.OE_A) / 100

# Market blending
df["Projection"] = 0.6 * df.Raw_Model + 0.4 * df.Market_Total

# Calibration to market
df["Projection"] -= (df.Projection.mean() - df.Market_Total.mean())

# ============================================================
# LATE-SEASON VARIANCE REDUCTION (FEATURE #2)
# ============================================================
month = datetime.now().month
season_factor = 0.85 if month >= 2 else 1.0

df["STD"] = (9 + (df.Tempo / 75) * 4) * season_factor

# ============================================================
# DIRECTIONAL MODELING (FEATURE #3)
# ============================================================
df["Prob_Over"] = df.apply(
    lambda r: prob_over(r.Market_Total, r.Projection, r.STD), axis=1
)
df["Prob_Under"] = df.apply(
    lambda r: prob_under(r.Market_Total, r.Projection, r.STD), axis=1
)

df["Side"] = np.where(
    df.Prob_Over > df.Prob_Under, "OVER", "UNDER"
)

df["Prob"] = df[["Prob_Over","Prob_Under"]].max(axis=1)
df["Edge"] = (df.Prob - implied_prob(ODDS)) * 100
df["Kelly_%"] = df.Prob.apply(lambda p: kelly_fraction(p, ODDS) * 100)

# ============================================================
# CLV CONFIRMATION (FEATURE #4)
# ============================================================
now = datetime.now()

for _, r in df.iterrows():
    line_history.loc[len(line_history)] = [
        r.Game, r.Market_Total, now
    ]

line_history.to_csv(LINE_HISTORY_FILE, index=False)

clv = line_history.groupby("Game").Line.diff()

df["CLV_OK"] = df.Game.map(
    lambda g: clv.loc[clv.index.get_level_values(0) == g].mean()
)

df = df[
    ((df.Side == "OVER") & (df.CLV_OK > 0)) |
    ((df.Side == "UNDER") & (df.CLV_OK < 0))
]

# ============================================================
# FILTERS
# ============================================================
min_prob = st.slider("Minimum Probability %", 52, 70, 57)
min_edge = st.slider("Minimum Edge %", 0, 10, 2)

bets = df[
    (df.Prob * 100 >= min_prob) &
    (df.Edge >= min_edge)
]

# ============================================================
# DISPLAY BETS
# ============================================================
st.subheader("📊 Qualified Bets")

st.dataframe(
    bets[[
        "Game","Side","Market_Total","Projection",
        "Prob","Edge","Kelly_%"
    ]].assign(Prob=lambda x:(x.Prob*100).round(1)),
    use_container_width=True
)

# ============================================================
# LOG RESULTS
# ============================================================
st.subheader("📝 Log Results")

for i, r in bets.iterrows():
    c1, c2, c3 = st.columns([3,1,1])
    c1.write(f"{r.Game} — {r.Side}")

    units = round(BANKROLL * r["Kelly_%"] / 100 / 100, 2)

    if c2.button("WIN", key=f"w{i}"):
        bet_log.loc[len(bet_log)] = [
            r.Game, r.Side, r.Market_Total,
            r.Prob, r.Edge, units * 0.91, "W", now
        ]

    if c3.button("LOSS", key=f"l{i}"):
        bet_log.loc[len(bet_log)] = [
            r.Game, r.Side, r.Market_Total,
            r.Prob, r.Edge, -units, "L", now
        ]

bet_log.to_csv(BET_LOG_FILE, index=False)

# ============================================================
# PERFORMANCE SUMMARY
# ============================================================
st.subheader("📈 Performance Summary")

total = len(bet_log)
wins = (bet_log.Result == "W").sum()
units = bet_log.Units.sum()

roi = (units / total * 100) if total else 0
win_pct = (wins / total * 100) if total else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Bets", total)
c2.metric("Win %", round(win_pct,1))
c3.metric("Units", round(units,2))
c4.metric("ROI %", round(roi,2))
