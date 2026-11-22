import pandas as pd
from pathlib import Path

def  prepare_data(keep_dates=False):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_PATH = PROJECT_ROOT / "data" / "aapl_prices.csv"

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["Date"],
        index_col="Date"
    )

    df["return"]=df["Close"].pct_change()
    df["return_1d"]=df["return"]
    df["return_mean_3d"]=df["return"].rolling(3).mean()
    df["return_mean_5d"]=df["return"].rolling(5).mean()
    df["volatility_5d"]=df["return"].rolling(5).std()
    df["target"]=df["return"].shift(-1)
    df=df.dropna()
    feature_cols=[
        "return_1d",
        "return_mean_3d",
        "return_mean_5d",
        "volatility_5d"
    ]
    if keep_dates:
        X = df[feature_cols]
    else:
        X = df[feature_cols].to_numpy()

    y = df["target"].to_numpy()
    split=int(len(X)*0.8)
    X_train=X[:split]
    y_train=y[:split]
    X_test=X[split:]
    y_test=y[split:]
    return X_train,y_train,X_test,y_test