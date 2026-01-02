import pandas as pd
import requests
from datetime import datetime

URL = "https://barttorvik.com/trank.php"
HEADERS = {"User-Agent": "Mozilla/5.0"}
OUTPUT = "team_stats.csv"

def scrape():
    r = requests.get(URL, headers=HEADERS, timeout=20)
    r.raise_for_status()

    df = pd.read_html(r.text)[0]

    df = df[["Team", "AdjOE", "AdjDE", "Tempo"]]

    df["Team"] = (
        df["Team"]
        .str.replace(r"\d+", "", regex=True)
        .str.strip()
    )

    df.to_csv(OUTPUT, index=False)
    print(f"Updated {OUTPUT} @ {datetime.now()}")

if __name__ == "__main__":
    scrape()
