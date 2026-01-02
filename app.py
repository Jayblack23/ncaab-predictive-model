# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import pandas as pd
import requests
import os
from math import erf, sqrt
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Elite NCAAB Totals Model", layout="wide")

BANKROLL = 1000
ODDS = -110
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

BET_LOG_FILE = f"{DATA_DIR}/bet_log.csv"
LINE_HISTORY_FILE = f"{DATA_DIR}/line_history.csv"

# ============================================================
# INIT STORAGE
# ============================================================
if not os.path.exists(BET_LOG_FILE):
    pd.DataFrame(columns=[
        "Game","Side","Bet_Line","Close_Line",
        "Prob","Edge","Units","Result","Date"
    ]).to_csv(BET_LOG_FILE, index=False)

if not os.path.exists(LINE_HISTORY_FILE):
    pd.DataFrame(columns=["Game","Line","Time"]).to_csv(
        LINE_HISTORY_FILE, index=False
    )

bet_log = pd.read_csv(BET_LOG_FILE)
line_history = pd.read_csv(LINE_HISTORY_FILE)

# ============================================================
# MATH
# ============================================================
def normal_cdf(x):
    return (1 + erf(x / sqrt(2))) / 2

def implied_prob(odds):
    return abs(odds) / (abs(odds) + 100)

def fair_odds(p):
    return round(-(p/(1-p))*100) if p > 0.5 else round((1-p)/p*100)

def prob_over(line, proj, std):
    z = (line - proj) / std
    return 1 - normal_cdf(z)

def prob_under(line, proj, std):
    return normal_cdf((line - proj) / std)

def kelly(p, odds):
    b = 100 / abs(odds)
    return max((p*b - (1-p))/b, 0)

# ============================================================
# FETCH ODDS
# ============================================================
@st.cache_data(ttl=300)
def fetch_odds():
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
            total = g["bookmakers"][0]["markets"][0]["outcomes"][0]["point"]
            rows.append({
                "Game": f"{g['away_team']} vs {g['home_team']}",
                "Market_Total": total
            })
        except:
            continue

    return pd.DataFrame(rows)

# ============================================================
# UI
# ============================================================
st.title("🏀 Elite NCAAB Totals Model")
st.caption("Directional • Calibrated • CLV-Aware")

if st.button("🔄 Refresh Odds"):
    st.experimental_rerun()

df = fetch_odds()
if df.empty:
    st.error("No live odds available.")
    st.stop()

# ============================================================
# UPLOAD REAL TEAM DATA (CRITICAL UPGRADE)
# ============================================================
st.subheader("📂 Upload Team Tempo & Efficiency CSV")
st.caption("Columns required: Team, Tempo, AdjOE")

team_file = st.file_uploader("Upload CSV", type="csv")

if team_file is None:
    st.warning("Upload required to proceed.")
    st.stop()

teams = pd.read_csv(team_file)

def get_team_stats(game, col):
    away, home = game.split(" vs ")
    try:
        return (
            teams.loc[teams.Team==home, col].values[0],
            teams.loc[teams.Team==away, col].values[0]
        )
    except:
        return None, None

df["Tempo_H"], df["Tempo_A"] = zip(*df.Game.map(lambda g: get_team_stats(g,"Tempo")))
df["AdjOE_H"], df["AdjOE_A"] = zip(*df.Game.map(lambda g: get_team_stats(g,"AdjOE")))

df.dropna(inplace=True)

# ============================================================
# PROJECTION ENGINE
# ============================================================
df["Tempo"] = (df["Tempo_H"] + df["Tempo_A"]) / 2
df["Raw_Model_Total"] = df["Tempo"] * (df["AdjOE_H"] + df["AdjOE_A"]) / 100

# Market blend
df["Projection"] = 0.6 * df["Raw_Model_Total"] + 0.4 * df["Market_Total"]

# Calibration
df["Projection"] -= (df["Projection"].mean() - df["Market_Total"].mean())

# ============================================================
# LATE-SEASON VARIANCE REDUCTION (UPGRADE)
# ============================================================
month = datetime.now().month
season_factor = 0.85 if month >= 2 else 1.0
df["STD"] = (9 + (df["Tempo"]/75)*4) * season_factor

# ============================================================
# DIRECTIONAL MODELING (UPGRADE)
# ============================================================
df["Prob_Over"] = df.apply(
    lambda r: prob_over(r.Market_Total, r.Projection, r.STD), axis=1
)
df["Prob_Under"] = df.apply(
    lambda r: prob_under(r.Market_Total, r.Projection, r.STD), axis=1
)

df["Best_Side"] = df.apply(
    lambda r: "OVER" if r.Prob_Over > r.Prob_Under else "UNDER", axis=1
)

df["Prob"] = df[["Prob_Over","Prob_Under"]].max(axis=1)
df["Edge"] = (df["Prob"] - implied_prob(ODDS)) * 100
df["Kelly_%"] = df["Prob"].apply(lambda p: kelly(p, ODDS)*100)

# ============================================================
# CLV AUTO-LOCK (UPGRADE)
# ============================================================
now = datetime.now()
for _, r in df.iterrows():
    line_history.loc[len(line_history)] = [r.Game, r.Market_Total, now]
line_history.to_csv(LINE_HISTORY_FILE, index=False)

recent = line_history.groupby("Game").tail(2)
clv_check = recent.groupby("Game").Line.diff()

df["CLV_Confirmed"] = df.Game.map(
    lambda g: clv_check.loc[clv_check.index.get_level_values(0)==g].mean()
)

df["CLV_OK"] = (
    ((df.Best_Side=="OVER") & (df.CLV_Confirmed > 0)) |
    ((df.Best_Side=="UNDER") & (df.CLV_Confirmed < 0))
)

# ============================================================
# FILTERS
# ============================================================
min_prob = st.slider("Min Probability %", 52, 70, 57)
min_edge = st.slider("Min Edge %", 0, 10, 2)

bets = df[
    (df.Prob*100 >= min_prob) &
    (df.Edge >= min_edge) &
    (df.CLV_OK)
]

# ============================================================
# DISPLAY
# ============================================================
st.subheader("📊 Qualified Bets (CLV-Confirmed)")
st.dataframe(
    bets[[
        "Game","Best_Side","Market_Total","Projection",
        "Prob","Edge","Kelly_%"
    ]].assign(Prob=lambda x:(x.Prob*100).round(1)),
    use_container_width=True
)

# ============================================================
# LOG RESULTS
# ============================================================
st.subheader("📝 Log Results")

for i, r in bets.iterrows():
    c1,c2,c3 = st.columns([3,1,1])
    c1.write(f"{r.Game} — {r.Best_Side}")

    units = round(BANKROLL * r["Kelly_%"]/100/100, 2)

    if c2.button("WIN", key=f"w{i}"):
        bet_log.loc[len(bet_log)] = [
            r.Game, r.Best_Side, r.Market_Total, None,
            r.Prob, r.Edge, units*0.91, "W", now
        ]

    if c3.button("LOSS", key=f"l{i}"):
        bet_log.loc[len(bet_log)] = [
            r.Game, r.Best_Side, r.Market_Total, None,
            r.Prob, r.Edge, -units, "L", now
        ]

bet_log.to_csv(BET_LOG_FILE, index=False)

# ============================================================
# PERFORMANCE
# ============================================================
st.subheader("📈 Performance")

total = len(bet_log)
wins = (bet_log.Result=="W").sum()
units = bet_log.Units.sum()

roi = (units/total*100) if total else 0
win_pct = (wins/total*100) if total else 0

c1,c2,c3,c4 = st.columns(4)
c1.metric("Bets", total)
c2.metric("Win %", round(win_pct,1))
c3.metric("Units", round(units,2))
c4.metric("ROI %", round(roi,2))
