import pandas as pd
from pathlib import Path


def load_oil():
    path = Path("data/raw/oil_prices.csv")
    df = pd.read_csv(path)
    return df

def clean_oil(df):

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset="Date")

    return df

def create_oil_features(df):

    df["oil_return"] = df["oil_price"].pct_change()
    df["oil_lag_1"] = df["oil_price"].shift(1)
    df["oil_lag_3"] = df["oil_price"].shift(3)

    df = df.dropna()

    return df

def save_oil(df):

    path = Path("data/processed/oil_features.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)
    print(f"Oil dataset saved to {path}")

def main():

    df = load_oil()
    df = clean_oil(df)
    df = create_oil_features(df)
    save_oil(df)

if __name__ == "__main__":
    main()