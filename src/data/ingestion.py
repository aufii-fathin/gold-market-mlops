import yfinance as yf
import pandas as pd
from pathlib import Path
from fredapi import Fred
from dotenv import load_dotenv
import os

load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY")

RAW_PATH = Path("data/raw")
RAW_PATH.mkdir(parents=True, exist_ok=True)


def fetch_gold_data(period="10y"):
    """
    Fetch historical gold price data from Yahoo Finance.
    """
    ticker = "GC=F"

    df = yf.download(
        ticker,
        period=period,
        interval="1d"
    )

    # handle multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    return df


def fetch_oil_data(period="10y"):

    ticker = "CL=F"

    df = yf.download(ticker, period=period, interval="1d")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()[["Date", "Close"]]
    df = df.rename(columns={"Close": "oil_price"})

    return df


def fetch_macro_data(start_date):

    fred = Fred(api_key=FRED_API_KEY)

    cpi = fred.get_series("CPIAUCSL")

    fed_rate = fred.get_series("DFF")
    usd_index = fred.get_series("DTWEXBGS")

    cpi_df = cpi.to_frame(name="cpi")
    fed_df = fed_rate.to_frame(name="fed_rate")
    usd_df = usd_index.to_frame(name="usd_index")

    macro_df = cpi_df.join(fed_df, how="outer")
    macro_df = macro_df.join(usd_df, how="outer")

    macro_df = macro_df.reset_index()
    macro_df = macro_df.rename(columns={"index": "Date"})

    macro_df["Date"] = pd.to_datetime(macro_df["Date"])

    macro_df = macro_df[macro_df["Date"] >= start_date]

    return macro_df


def save_dataset(df, filename):

    path = RAW_PATH / filename
    df.to_csv(path, index=False)

    print(f"Saved → {path}")


def main():

    print("Fetching gold data...")
    gold_df = fetch_gold_data()
    save_dataset(gold_df, "gold_prices.csv")

    print("Fetching oil data...")
    oil_df = fetch_oil_data()
    save_dataset(oil_df, "oil_prices.csv")

    start_date = gold_df["Date"].min()

    print("Fetching macro data from FRED...")
    macro_df = fetch_macro_data(start_date)
    save_dataset(macro_df, "macro_fred.csv")

    print("All ingestion complete.")


if __name__ == "__main__":
    main()