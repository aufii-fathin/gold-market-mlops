import pandas as pd
from pathlib import Path

def load_datasets():

    gold = pd.read_csv("data/processed/gold_features.csv")
    oil = pd.read_csv("data/processed/oil_features.csv")
    macro = pd.read_csv("data/processed/fred_features.csv")

    return gold, oil, macro

def merge_all(gold, oil, macro):

    gold["Date"] = pd.to_datetime(gold["Date"])
    oil["Date"] = pd.to_datetime(oil["Date"])
    macro["Date"] = pd.to_datetime(macro["Date"])

    df = gold.merge(oil, on="Date", how="left")
    df = df.merge(macro, on="Date", how="left")

    # forward fill macro
    df = df.ffill()

    return df

def save_dataset(df):

    path = Path("data/processed/market_dataset.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)
    print(f"Merged dataset saved to {path}")

def main():

    gold, oil, macro = load_datasets()
    df = merge_all(gold, oil, macro)
    save_dataset(df)

if __name__ == "__main__":
    main()