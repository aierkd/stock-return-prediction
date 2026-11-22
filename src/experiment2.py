from features import prepare_data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

X_train, y_train, X_test, y_test = prepare_data()
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
print(search.best_params_)
print(-search.best_score_)
y_pred=search.predict(X_test)
model_mse=mean_squared_error(y_test,y_pred)
baseline=np.zeros_like(y_test)
baseline_mse=mean_squared_error(y_test,baseline)
oos_r2=1-model_mse/baseline_mse
print("Model MSE",model_mse)
print("Baseline MSE",baseline_mse)
print("OOS R2",oos_r2)
# Alpha:1000
# OOS R2:-0.00861