import matplotlib.pyplot as plt
from pathlib import Path
from features import prepare_data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
#Figure1：
X_train, y_train, X_test, y_test = prepare_data(keep_dates=True)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1000))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Test observations:", len(y_test))
print("Predictions:", len(y_pred))
print("First test date:", X_test.index[0])

plt.figure(figsize=(12, 5))
plt.plot(X_test.index, y_test, label="Actual Returns", alpha=0.7)
plt.plot(X_test.index, y_pred, label="Predicted Returns", alpha=0.9)
plt.title("AAPL: Actual vs Predicted Daily Returns")
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
PROJECT_ROOT=Path(__file__).resolve().parents[1]
FIGURE_DIR=PROJECT_ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(
    FIGURE_DIR / "actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)
# plt.show()


#Figure2
import numpy as np

alphas=[0.01, 0.1, 1, 10, 100, 1000]

feature_names = [
    "return_1d",
    "return_mean_3d",
    "return_mean_5d",
    "volatility_5d"
]

coefficient_history=[]

for alpha in alphas:

    ridge_model=Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha))
    ])

    ridge_model.fit(X_train, y_train)

    coefficients=ridge_model.named_steps["ridge"].coef_

    coefficient_history.append(coefficients)
coefficient_history=np.array(coefficient_history)
print("Coefficient matrix shape:", coefficient_history.shape)
print(coefficient_history)
plt.figure(figsize=(10, 6))
for i, feature in enumerate(feature_names):

    plt.plot(
        alphas,
        coefficient_history[:, i],
        marker="o",
        label=feature
    )

plt.xscale("log")

plt.axhline(y=0, color="gray", linestyle="--")

plt.title("Ridge Coefficients vs Regularization Strength")
plt.xlabel("Alpha (log scale)")
plt.ylabel("Standardized Coefficient")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "ridge_coefficients.png",
    dpi=300,
    bbox_inches="tight"
)
# plt.show()



#Figure3
import pandas as pd
from sklearn.metrics import mean_squared_error

results=pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
}, index=X_test.index)

results["Year"]=results.index.year

years=[]
yearly_oos_r2=[]

for year, group in results.groupby("Year"):
    actual=group["Actual"].to_numpy()
    predicted=group["Predicted"].to_numpy()
    model_mse=mean_squared_error(actual, predicted)
    baseline_mse=mean_squared_error(
        actual, np.zeros_like(actual)
    )

    oos_r2=1 - model_mse / baseline_mse

    years.append(str(year))
    yearly_oos_r2.append(oos_r2 * 100)

    print(f"{year}:OOS R2={oos_r2:.4%}")

plt.figure(figsize=(9, 5))
plt.bar(years, yearly_oos_r2)
plt.axhline(y=0, color="black", linestyle="--")
plt.title("AAPL: Year-by-Year Out-of-Sample R2")
plt.xlabel("Year")
plt.ylabel("OOS R2 (%)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "yearly_oos_r2.png",
    dpi=300,
    bbox_inches="tight"
)





#Figure4
X_all=pd.concat([X_train, X_test])
y_all=np.concatenate([y_train, y_test])
initial_train_size=len(X_train)
step_size=60
walk_actual=[]
walk_predicted=[]
for start in range(initial_train_size, len(X_all), step_size):

    end=min(start + step_size, len(X_all))

    X_tr=X_all.iloc[:start]
    y_tr=y_all[:start]

    X_te=X_all.iloc[start:end]
    y_te=y_all[start:end]

    walk_model=Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1000))
    ])
    walk_model.fit(X_tr, y_tr)
    predictions=walk_model.predict(X_te)
    walk_actual.extend(y_te)
    walk_predicted.extend(predictions)

walk_actual=np.array(walk_actual)
walk_predicted=np.array(walk_predicted)
fixed_mse=mean_squared_error(y_test, y_pred)
walk_mse=mean_squared_error(
    walk_actual, walk_predicted
)
baseline_mse=mean_squared_error(
    y_test, np.zeros_like(y_test)
)
fixed_oos_r2=1-fixed_mse/baseline_mse
walk_oos_r2=1-walk_mse/baseline_mse
print("Fixed OOS R2:", fixed_oos_r2)
print("Walk-Forward OOS R2:", walk_oos_r2)
methods=["Fixed Ridge", "Walk-Forward Ridge"]
scores=[
    fixed_oos_r2 * 100,
    walk_oos_r2 * 100
]
plt.figure(figsize=(9, 5))
plt.bar(methods, scores)
plt.axhline(y=0, color="black", linestyle="--")
plt.title("Fixed vs Walk-Forward Ridge Regression")
plt.ylabel("Out-of-Sample R2 (%)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "walk_forward_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()