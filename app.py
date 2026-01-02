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
# PAGE CONFIG (MOBILE FIRST)
# ============================================================
st.set_page_config(
    page_title="Elite NCAAB Totals",
    layout="wide",
    initial_sidebar_state="collapsed"
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
WEIGHTS_FILE = f"{DATA_DIR}/weights.csv"

# ============================================================
# INIT STORAGE
# ============================================================
if not os.path.exists(BET_LOG_FILE):
    pd.DataFrame(columns=[
        "Game","Side","Open_Line","Close_Line",
        "Prob","Edge","Units","Result","Date"
    ]).to_csv(BET_LOG_FILE, index=False)

if not os.path.exists(LINE_HISTORY_FILE):
    pd.DataFrame(columns=["Game","Line","Time"]).to_csv(
        LINE_HISTORY_FILE, index=False
    )

if not os.path.exists(WEIGHTS_FILE):
    pd.DataFrame([{
        "model_weight": 0.6,
        "market_weight": 0.4
    }]).to_csv(WEIGHTS_FILE, index=False)

bet_log = pd.read_csv(BET_LOG_FILE)
line_history = pd.read_csv(LINE_HISTORY_FILE)
weights = pd.read_csv(WEIGHTS_FILE)

MODEL_WEIGHT = weights.loc[0,"model_weight"]
MARKET_WEIGHT = weights.loc[0,"market_weight"]

# ============================================================
# MATH HELPERS
# ============================================================
def normal_cdf(x):
    return (1 + erf(x / sqrt(2))) / 2

def implied_prob(odds):
    return abs(odds) / (abs(odds) + 100)

def prob_over(line, proj, std):
    return 1 - normal_cdf((line - proj) / std)

def prob_under(line, proj, std):
    return normal_cdf((line - proj) / std)

def kelly(p, odds):
    b = 100 / abs(odds)
    return max((p*b - (1-p))/b, 0)

# ============================================================
# FETCH MULTI-BOOK TOTALS (LINE SHOPPING)
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

    for g in data:
        try:
            totals = []
            for b in g["bookmakers"]:
                for o in b["markets"][0]["outcomes"]:
                    totals.append(o["point"])

            rows.append({
                "Game": f"{g['away_team']} vs {g['home_team']}",
                "Best_Total": max(totals),
                "Worst_Total": min(totals),
                "Market_Total": np.mean(totals)
            })
        except:
            continue

    return pd.DataFrame(rows)

# ============================================================
# HEADER
# ============================================================
st.title("🏀 Elite NCAAB Totals Model")
st.caption("CLV • Line Shopping • Adaptive Weights • Mobile Optimized")

if st.button("🔄 Refresh Odds"):
    st.experimental_rerun()

df = fetch_odds()
if df.empty:
    st.error("No odds available.")
    st.stop()

# ============================================================
# TEAM DATA UPLOAD
# ============================================================
st.subheader("📂 Upload Team Metrics")
st.caption("Required columns: Team, Tempo, AdjOE")

team_file = st.file_uploader("Upload CSV", type="csv")
if team_file is None:
    st.stop()

teams = pd.read_csv(team_file)

def team_stat(game, col):
    away, home = game.split(" vs ")
    try:
        return (
            teams.loc[teams.Team==home, col].values[0],
            teams.loc[teams.Team==away, col].values[0]
        )
    except:
        return None, None

df["Tempo_H"], df["Tempo_A"] = zip(*df.Game.map(lambda g: team_stat(g,"Tempo")))
df["OE_H"], df["OE_A"] = zip(*df.Game.map(lambda g: team_stat(g,"AdjOE")))
df.dropna(inplace=True)

# ============================================================
# MODEL
# ============================================================
df["Tempo"] = (df.Tempo_H + df.Tempo_A) / 2
df["Raw_Model"] = df.Tempo * (df.OE_H + df.OE_A) / 100

df["Projection"] = (
    MODEL_WEIGHT * df.Raw_Model +
    MARKET_WEIGHT * df.Market_Total
)

# Calibration
df["Projection"] -= (df.Projection.mean() - df.Market_Total.mean())

# Late season variance tightening
month = datetime.now().month
season_factor = 0.85 if month >= 2 else 1.0
df["STD"] = (9 + (df.Tempo/75)*4) * season_factor

# ============================================================
# DIRECTIONAL PROBABILITIES
# ============================================================
df["Prob_Over"] = df.apply(lambda r: prob_over(r.Market_Total, r.Projection, r.STD), axis=1)
df["Prob_Under"] = df.apply(lambda r: prob_under(r.Market_Total, r.Projection, r.STD), axis=1)

df["Side"] = np.where(df.Prob_Over > df.Prob_Under, "OVER", "UNDER")
df["Prob"] = df[["Prob_Over","Prob_Under"]].max(axis=1)
df["Edge"] = (df.Prob - implied_prob(ODDS)) * 100
df["Kelly_%"] = df.Prob.apply(lambda p: kelly(p, ODDS) * 100)

# ============================================================
# CLV TRACKING
# ============================================================
now = datetime.now()
for _, r in df.iterrows():
    line_history.loc[len(line_history)] = [r.Game, r.Market_Total, now]

line_history.to_csv(LINE_HISTORY_FILE, index=False)

clv = line_history.groupby("Game").Line.diff()
df["CLV_OK"] = df.Game.map(lambda g: clv.loc[clv.index.get_level_values(0)==g].mean())

df = df[
    ((df.Side=="OVER") & (df.CLV_OK > 0)) |
    ((df.Side=="UNDER") & (df.CLV_OK < 0))
]

# ============================================================
# FILTERS
# ============================================================
min_prob = st.slider("Min Probability %", 52, 70, 57)
min_edge = st.slider("Min Edge %", 0, 10, 2)

bets = df[
    (df.Prob*100 >= min_prob) &
    (df.Edge >= min_edge)
]

# ============================================================
# DISPLAY
# ============================================================
st.subheader("📊 Qualified Bets")

st.dataframe(
    bets[[
        "Game","Side","Market_Total","Best_Total","Worst_Total",
        "Projection","Prob","Edge","Kelly_%"
    ]].assign(Prob=lambda x:(x.Prob*100).round(1)),
    use_container_width=True
)

# ============================================================
# LOG RESULTS
# ============================================================
st.subheader("📝 Log Results")

for i, r in bets.iterrows():
    c1,c2,c3 = st.columns([3,1,1])
    c1.write(f"{r.Game} — {r.Side}")

    units = round(BANKROLL * r["Kelly_%"] / 100 / 100, 2)

    if c2.button("WIN", key=f"w{i}"):
        bet_log.loc[len(bet_log)] = [
            r.Game, r.Side, r.Market_Total, None,
            r.Prob, r.Edge, units*0.91, "W", now
        ]

    if c3.button("LOSS", key=f"l{i}"):
        bet_log.loc[len(bet_log)] = [
            r.Game, r.Side, r.Market_Total, None,
            r.Prob, r.Edge, -units, "L", now
        ]

bet_log.to_csv(BET_LOG_FILE, index=False)

# ============================================================
# PERFORMANCE + AUTO WEIGHT ADJUSTMENT
# ============================================================
st.subheader("📈 Performance")

total = len(bet_log)
wins = (bet_log.Result=="W").sum()
units = bet_log.Units.sum()

roi = (units/total*100) if total else 0
win_pct = (wins/total*100) if total else 0

if total >= 50:
    if roi < 0:
        MODEL_WEIGHT = max(MODEL_WEIGHT - 0.05, 0.4)
    else:
        MODEL_WEIGHT = min(MODEL_WEIGHT + 0.05, 0.75)

    MARKET_WEIGHT = 1 - MODEL_WEIGHT
    pd.DataFrame([{
        "model_weight": MODEL_WEIGHT,
        "market_weight": MARKET_WEIGHT
    }]).to_csv(WEIGHTS_FILE, index=False)

c1,c2,c3,c4 = st.columns(4)
c1.metric("Bets", total)
c2.metric("Win %", round(win_pct,1))
c3.metric("Units", round(units,2))
c4.metric("ROI %", round(roi,2))

# ============================================================
# CLV CHART
# ============================================================
st.subheader("📉 Closing Line Value Trend")

if len(line_history) > 5:
    clv_df = line_history.groupby("Game").Line.diff().dropna()
    st.line_chart(clv_df.reset_index(drop=True))
