import yfinance as yf

df=yf.download(
    "AAPL",
    start="2015-01-01",
    end="2026-01-01",
    auto_adjust=True
)
prices=df[[("Close","AAPL")]].copy()
prices.columns=["Close"]
print("Data shape:", prices.shape)
print("Missing values:")
print(prices.isna().sum())
print(prices.head())


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "aapl_prices.csv"

DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

prices.to_csv(DATA_PATH, index_label="Date")

print("Data saved to:", DATA_PATH)

