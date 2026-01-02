import streamlit as st
import pandas as pd
from math import erf, sqrt

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
final_df = pd.DataFrame(
    filtered,
    columns=["Game", "Market Total", "Model Total", "Over %", "Fair Odds", "Edge", "Decision"]
)
)

st.dataframe(final_df, use_container_width=True)
