import yfinance as yf
import pandas as pd


def fetch_gold_data(period="1y"):
    """
    Fetch historical gold price data (XAU/USD proxy: GC=F)
    """
    ticker = "GC=F"  # Gold Futures
    data = yf.download(ticker, period=period, interval="1d")

    return data


if __name__ == "__main__":
    df = fetch_gold_data()
    print("Data fetched successfully.")
    print(df.head())