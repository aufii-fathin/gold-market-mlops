import pandas as pd
from pathlib import Path


def load_macro():
    path = Path("data/raw/macro_fred.csv")
    df = pd.read_csv(path)
    return df

def clean_macro(df):

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # macro data berlaku sampai update berikutnya
    df = df.ffill()

    return df

def create_macro_features(df):

    df["usd_return"] = df["usd_index"].pct_change()
    df["rate_change"] = df["fed_rate"].diff()
    df["cpi_change"] = df["cpi"].pct_change()

    # real interest rate
    df["real_rate"] = df["fed_rate"] - df["cpi_change"]
    df = df.dropna()

    return df

def save_macro(df):

    path = Path("data/processed/fred_features.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)
    print(f"Macro dataset saved to {path}")

def main():

    df = load_macro()
    df = clean_macro(df)
    df = create_macro_features(df)
    save_macro(df)

if __name__ == "__main__":
    main()