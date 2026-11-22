from features import prepare_data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import pandas as pd

X_train, y_train, X_test, y_test = prepare_data(keep_dates=True)
model=Pipeline([
    ("scaler",StandardScaler()),
    ("Ridge",Ridge())
])
param_grid={
    "Ridge__alpha":[0.01, 0.1, 1, 10, 100, 1000]
}
search=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=TimeSeriesSplit(n_splits=5),
    scoring='neg_mean_squared_error'
)
search.fit(X_train,y_train)
y_pred=search.predict(X_test)
results=pd.DataFrame({
    "Actual":y_test,
    "Predicted":y_pred},
    index=X_test.index
)
results["Year"] = results.index.year
for year,group in results.groupby("Year"):
    actual=group["Actual"]
    prd=group["Predicted"]
    model_mse=mean_squared_error(actual,prd)
    baseline=np.zeros_like(actual)
    baseline_mse=mean_squared_error(actual, baseline)
    oos_r2=1 - model_mse / baseline_mse
    print(f"\nYear: {year}")
    print(f"Number of observations: {len(group)}")
    print(f"Model MSE: {model_mse:.8f}")
    print(f"Baseline MSE: {baseline_mse:.8f}")
    print(f"OOS R2: {oos_r2:.4%}")
# Year: 2023
# Number of observations: 51
# Model MSE: 0.00012970
# Baseline MSE: 0.00012721
# OOS R2: -1.9568%
#
# Year: 2024
# Number of observations: 252
# Model MSE: 0.00020346
# Baseline MSE: 0.00020232
# OOS R2: -0.5598%
#
# Year: 2025
# Number of observations: 249
# Model MSE: 0.00042090
# Baseline MSE: 0.00041698
# OOS R2: -0.9412%
X_all=pd.concat([X_train,X_test])
y_all=np.concatenate([y_train,y_test])
intial_train_size=len(X_train)
step_size=60
walk_actual=[]
walk_predicted=[]
for start in range(intial_train_size,len(X_all),step_size):
    end=min(start+step_size,len(X_all))
    X_tr=X_all.iloc[:start]
    y_tr=y_all[:start]
    X_te=X_all.iloc[start:end]
    y_te=y_all[start:end]
    walk_model=Pipeline([
        ("scaler",StandardScaler()),
        ("Ridge",Ridge(alpha=1000))
    ])
    walk_model.fit(X_tr,y_tr)
    predictions=walk_model.predict(X_te)
    walk_actual.extend(y_te)
    walk_predicted.extend(predictions)

walk_actual=np.array(walk_actual)
walk_predicted=np.array(walk_predicted)
walk_mse=mean_squared_error(walk_actual,walk_predicted)
walk_baseline_mse=mean_squared_error(walk_actual,np.zeros_like(walk_actual))
walk_oos_r2=1-walk_mse/walk_baseline_mse
print("Walk MSE:",walk_mse)
print("Baseline MSE:",walk_baseline_mse)
print("OOS R2:",walk_oos_r2)
# Walk MSE: 0.00029441964901701366
# Baseline MSE: 0.00029221135215531653
# OOS R2: -0.007557190524628776






















