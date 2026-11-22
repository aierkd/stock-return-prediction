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
best_model=search.best_estimator_
ridge = best_model.named_steps["Ridge"]
coefficients=ridge.coef_
intercepts=ridge.intercept_
print("coefficients:",coefficients)
print("intercepts:",intercepts)
feature_names=[
    "return_1d",
    "return_mean_3d",
    "return_mean_5d",
    "volatility_5d"
]
for name,coefficient in zip(feature_names,coefficients):
    print(f"{name}: {coefficient:.8f}")
# coefficients: [-7.72104107e-04 -5.20756653e-04 -2.65103971e-05 -4.71885446e-04]
# intercepts: 0.0010518653953684959
# return_1d: -0.00077210
# return_mean_3d: -0.00052076
# return_mean_5d: -0.00002651
# volatility_5d: -0.00047189


for alpha in [0.01,0.1,1,10,100,1000]:
    model=Pipeline([
        ("scaler",StandardScaler()),
        ("ridge",Ridge(alpha=alpha))
    ])
    model.fit(X_train,y_train)
    coefficients=model.named_steps["ridge"].coef_
    print(f"\nAlpha = {alpha}")
    for name,coef in zip(feature_names,coefficients):
        print(f"{name}: {coef:.8f}")

# Alpha = 0.01
# return_1d: -0.00108464
# return_mean_3d: -0.00085169
# return_mean_5d: 0.00032289
# volatility_5d: -0.00065428
#
# Alpha = 0.1
# return_1d: -0.00108460
# return_mean_3d: -0.00085160
# return_mean_5d: 0.00032280
# volatility_5d: -0.00065427
#
# Alpha = 1
# return_1d: -0.00108424
# return_mean_3d: -0.00085080
# return_mean_5d: 0.00032193
# volatility_5d: -0.00065407
#
# Alpha = 10
# return_1d: -0.00108061
# return_mean_3d: -0.00084291
# return_mean_5d: 0.00031340
# volatility_5d: -0.00065214
#
# Alpha = 100
# return_1d: -0.00104440
# return_mean_3d: -0.00077639
# return_mean_5d: 0.00024068
# volatility_5d: -0.00063271
#
# Alpha = 1000
# return_1d: -0.00077210
# return_mean_3d: -0.00052076
# return_mean_5d: -0.00002651
# volatility_5d: -0.00047189





