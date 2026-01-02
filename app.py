import os
import streamlit as st
import pandas as pd
from math import erf, sqrt
BET_FILE = "bets.csv"

if not os.path.exists(BET_FILE):
    pd.DataFrame(columns=["Result"]).to_csv(BET_FILE, index=False)
if "bet_log" not in st.session_state:
    st.session_state.bet_log = []
# ---------- Helper functions ----------
def normal_cdf(x):
    return (1.0 + erf(x / sqrt(2.0))) / 2.0

def over_probability(line, projection, std=11):
    z = (line - projection) / std
    return round(1 - normal_cdf(z), 4)

def fair_odds(prob):
    if prob >= 0.5:
        return round(-(prob / (1 - prob)) * 100)
    else:
        return round((1 - prob) / prob * 100)

def implied_prob(odds):
    return abs(odds) / (abs(odds) + 100)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="NCAAB Totals Model", layout="wide")

st.title("🏀 NCAAB Predictive Totals Model")
st.caption("Custom ensemble model (KenPom-style + simulation logic)")

df = pd.read_csv("data.csv")

results = []

for _, row in df.iterrows():
    prob = over_probability(row["Line"], row["Projection"])
    odds = fair_odds(prob)
    edge = round((prob - implied_prob(-110)) * 100, 2)

    results.append([
        row["Game"],
        row["Line"],
        row["Projection"],
        f"{prob*100:.1f}%",
        odds,
        f"{edge}%"
    ])decision = "BET" if prob >= min_prob/100 and edge >= min_edge else "PASS"
min_prob = st.slider("Minimum Over Probability (%)", 50, 70, 58)
min_edge = st.slider("Minimum Edge (%)", 0, 10, 2)

filtered = [
    r for r in results
    if float(r[3].replace("%","")) >= min_prob and float(r[5].replace("%","")) >= min_edge
]
(final_df = final_df[
    (final_df["Over %"].str.replace("%","").astype(float) >= min_prob) &
    (final_df["Edge"].str.replace("%","").astype(float) >= min_edge)

    filtered,
    columns=["Game", "Market Total", "Model Total", "Over %", "Fair Odds", "Edge", "Decision"]
)
)

st.dataframe(final_df, use_container_width=True)
st.subheader("📊 Log Bet Results")

for i, row in final_df.iterrows():
    if row["Decision"] == "BET":
        col1, col2, col3 = st.columns([3,1,1])

        col1.write(row["Game"])

        if col2.button("WIN", key=f"win_{i}"):
            st.session_state.bet_log.append(1)

        if col3.button("LOSS", key=f"loss_{i}"):
            st.session_state.bet_log.append(-1) 
st.subheader("📈 Performance Summary")

total_bets = len(st.session_state.bet_log)
wins = st.session_state.bet_log.count(1)
losses = st.session_state.bet_log.count(-1)

units = sum(st.session_state.bet_log)

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
