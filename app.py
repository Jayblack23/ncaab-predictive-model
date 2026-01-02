# =========================
# IMPORTS (TOP OF FILE)
# =========================
import streamlit as st
import pandas as pd
import requests
import os
from math import erf, sqrt

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="NCAAB Predictive Totals Model", layout="wide")

BET_FILE = "bets.csv"

# =========================
# INITIALIZE BET STORAGE
# =========================
if not os.path.exists(BET_FILE):
    pd.DataFrame(columns=["Result"]).to_csv(BET_FILE, index=False)

bet_history = pd.read_csv(BET_FILE)

# =========================
# HELPER FUNCTIONS
# =========================
def normal_cdf(x):
    return (1.0 + erf(x / sqrt(2.0))) / 2.0

def over_probability(line, projection, std=11):
    z = (line - projection) / std
    return 1 - normal_cdf(z)

def fair_odds(prob):
    if prob >= 0.5:
        return round(-(prob / (1 - prob)) * 100)
    else:
        return round((1 - prob) / prob * 100)

def implied_prob(odds):
    return abs(odds) / (abs(odds) + 100)

# =========================
# FETCH LIVE TOTALS
# =========================
def fetch_ncaab_totals():
    api_key = st.secrets["ODDS_API_KEY"]

    url = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "totals",
        "oddsFormat": "american"
    }

    response = requests.get(url, params=params)

    st.subheader("🔍 Odds API Status")
    st.write("Status Code:", response.status_code)

    try:
        data = response.json()
    except:
        st.error("API response could not be parsed")
        return pd.DataFrame()

    if not isinstance(data, list):
        st.error("API did not return game list")
        st.write(data)
        return pd.DataFrame()

    games = []

    for game in data:
        try:
            bookmakers = game.get("bookmakers", [])
            if not bookmakers:
                continue

            markets = bookmakers[0].get("markets", [])
            if not markets:
                continue

            outcomes = markets[0].get("outcomes", [])
            if not outcomes:
                continue

            total = outcomes[0]["point"]

            games.append({
                "Game": f"{game['away_team']} vs {game['home_team']}",
                "Line": total
            })
        except:
            continue

    return pd.DataFrame(games)

# =========================
# UI HEADER
# =========================
st.title("🏀 NCAAB Predictive Totals Model")
st.caption("Live odds • Probability model • ROI tracking")

# =========================
# LOAD DATA
# =========================
st.subheader("🔄 Live Market Totals")

df = fetch_ncaab_totals()

if df.empty:
    st.warning("No live totals available — using manual fallback")
    df = pd.read_csv("data.csv")

# TEMP projection logic (replace later with true model)
df["Projection"] = df["Line"] + 2.5

# =========================
# FILTER CONTROLS
# =========================
min_prob = st.slider("Minimum Over Probability (%)", 50, 70, 58)
min_edge = st.slider("Minimum Edge (%)", 0, 10, 2)

# =========================
# MODEL CALCULATIONS
# =========================
results = []

for _, row in df.iterrows():
    prob = over_probability(row["Line"], row["Projection"])
    odds = fair_odds(prob)
    edge = round((prob - implied_prob(-110)) * 100, 2)

    decision = "BET" if prob >= min_prob / 100 and edge >= min_edge else "PASS"

    results.append({
        "Game": row["Game"],
        "Market Total": row["Line"],
        "Model Total": row["Projection"],
        "Over %": round(prob * 100, 1),
        "Fair Odds": odds,
        "Edge %": edge,
        "Decision": decision
    })

final_df = pd.DataFrame(results)

final_df = final_df[
    (final_df["Over %"] >= min_prob) &
    (final_df["Edge %"] >= min_edge)
]

# =========================
# DISPLAY TABLE
# =========================
st.subheader("📊 Model Results")
st.dataframe(final_df, use_container_width=True)

# =========================
# LOG BET RESULTS
# =========================
st.subheader("📝 Log Bet Results")

for i, row in final_df.iterrows():
    if row["Decision"] == "BET":
        col1, col2, col3 = st.columns([3, 1, 1])
        col1.write(row["Game"])

        if col2.button("WIN", key=f"win_{i}"):
            bet_history.loc[len(bet_history)] = [1]
            bet_history.to_csv(BET_FILE, index=False)

        if col3.button("LOSS", key=f"loss_{i}"):
            bet_history.loc[len(bet_history)] = [-1]
            bet_history.to_csv(BET_FILE, index=False)

# =========================
# PERFORMANCE SUMMARY
# =========================
st.subheader("📈 Performance Summary")

total_bets = len(bet_history)
wins = (bet_history["Result"] == 1).sum()
losses = (bet_history["Result"] == -1).sum()
units = bet_history["Result"].sum()

if total_bets > 0:
    roi = round((units / total_bets) * 100, 2)
    win_pct = round((wins / total_bets) * 100, 2)
else:
    roi = 0
    win_pct = 0

st.metric("Total Bets", total_bets)
st.metric("Wins", wins)
st.metric("Losses", losses)
st.metric("Units", units)
st.metric("ROI %", roi)
st.metric("Win %", win_pct)
