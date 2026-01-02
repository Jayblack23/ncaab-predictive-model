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
st.set_page_config(
    page_title="Elite NCAAB Totals Model",
    layout="wide"
)

BANKROLL = 1000
ODDS = -110

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

BET_LOG_FILE = f"{DATA_DIR}/bet_log.csv"
LINE_HISTORY_FILE = f"{DATA_DIR}/line_history.csv"
MODEL_WEIGHTS_FILE = f"{DATA_DIR}/model_weights.csv"

# ============================================================
# INITIALIZE FILES
# ============================================================
if not os.path.exists(BET_LOG_FILE):
    pd.DataFrame(columns=[
        "Game","Bet_Line","Close_Line",
        "Prob","Edge","Units","Result","Date"
    ]).to_csv(BET_LOG_FILE, index=False)

if not os.path.exists(LINE_HISTORY_FILE):
    pd.DataFrame(columns=["Game","Line","Time"]).to_csv(
        LINE_HISTORY_FILE, index=False
    )

if not os.path.exists(MODEL_WEIGHTS_FILE):
    pd.DataFrame({
        "signal": ["model", "market"],
        "weight": [0.65, 0.35]
    }).to_csv(MODEL_WEIGHTS_FILE, index=False)

bet_log = pd.read_csv(BET_LOG_FILE)
line_history = pd.read_csv(LINE_HISTORY_FILE)
weights = pd.read_csv(MODEL_WEIGHTS_FILE)

# ============================================================
# MATH
# ============================================================
def normal_cdf(x):
    return (1 + erf(x / sqrt(2))) / 2

def implied_prob(odds):
    return abs(odds) / (abs(odds) + 100)

def fair_odds(p):
    return round(-(p / (1 - p)) * 100) if p > 0.5 else round((1 - p) / p * 100)

def over_prob(line, proj, std):
    z = (line - proj) / std
    return 1 - normal_cdf(z)

def kelly_fraction(p, odds):
    b = 100 / abs(odds)
    return max((p * b - (1 - p)) / b, 0)

# ============================================================
# ODDS API
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
st.title("🏀 Elite NCAAB Totals Engine")
st.caption("Probability • CLV • Kelly • Self-Learning")

if st.button("🔄 Refresh Odds"):
    st.experimental_rerun()

df = fetch_odds()

if df.empty:
    st.warning("Using fallback data.csv")
    df = pd.read_csv("data.csv")

# ============================================================
# MODEL INPUTS (SAFE PLACEHOLDERS)
# Replace later with real KenPom CSV/API
# ============================================================
df["Tempo"] = 68 + (df.index % 6) * 2
df["AdjOE_H"] = 108 + (df.index % 4) * 1.5
df["AdjOE_A"] = 106 + (df.index % 4) * 1.5

# ============================================================
# PROJECTION ENGINE
# ============================================================
df["Model_Total"] = (
    df["Tempo"] * (df["AdjOE_H"] + df["AdjOE_A"]) / 100
)

w_model = weights.loc[weights.signal=="model","weight"].values[0]
w_market = weights.loc[weights.signal=="market","weight"].values[0]

df["Projection"] = (
    w_model * df["Model_Total"] +
    w_market * df["Market_Total"]
)

df["STD"] = 9 + (df["Tempo"] / 75) * 4

df["Prob"] = df.apply(
    lambda r: over_prob(r["Market_Total"], r["Projection"], r["STD"]),
    axis=1
)

df["Edge"] = (df["Prob"] - implied_prob(ODDS)) * 100
df["Fair_Odds"] = df["Prob"].apply(fair_odds)
df["Kelly_%"] = df["Prob"].apply(lambda p: kelly_fraction(p, ODDS) * 100)

# ============================================================
# FILTERS
# ============================================================
min_prob = st.slider("Min Probability %", 52, 70, 57)
min_edge = st.slider("Min Edge %", 0, 10, 2)

df["Decision"] = df.apply(
    lambda r: "BET"
    if r["Prob"]*100 >= min_prob and r["Edge"] >= min_edge
    else "PASS",
    axis=1
)

bets = df[df.Decision=="BET"].copy()

# ============================================================
# DISPLAY
# ============================================================
st.subheader("📊 Qualified Bets")
st.dataframe(
    bets[[
        "Game","Market_Total","Projection",
        "Prob","Edge","Fair_Odds","Kelly_%"
    ]].assign(Prob=lambda x:(x.Prob*100).round(1)),
    use_container_width=True
)

# ============================================================
# LINE HISTORY (STEAM / CLV)
# ============================================================
now = datetime.now()
for _, r in df.iterrows():
    line_history.loc[len(line_history)] = [r.Game, r.Market_Total, now]
line_history.to_csv(LINE_HISTORY_FILE, index=False)

# ============================================================
# LOG RESULTS
# ============================================================
st.subheader("📝 Log Results")

for i, r in bets.iterrows():
    c1,c2,c3 = st.columns([3,1,1])
    c1.write(r.Game)

    units = round(BANKROLL * r["Kelly_%"] / 100 / 100, 2)

    if c2.button("WIN", key=f"w{i}"):
        bet_log.loc[len(bet_log)] = [
            r.Game, r.Market_Total, None,
            r.Prob, r.Edge, units*0.91, "W", now
        ]

    if c3.button("LOSS", key=f"l{i}"):
        bet_log.loc[len(bet_log)] = [
            r.Game, r.Market_Total, None,
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

roi = (units / total * 100) if total else 0
win_pct = (wins / total * 100) if total else 0

c1,c2,c3,c4 = st.columns(4)
c1.metric("Bets", total)
c2.metric("Win %", round(win_pct,1))
c3.metric("Units", round(units,2))
c4.metric("ROI %", round(roi,2))
