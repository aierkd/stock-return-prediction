from features import prepare_data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.metrics import mean_squared_error

X_train, y_train, X_test, y_test = prepare_data()
model=Pipeline([
    ("scaler",StandardScaler()),
    ("Ridge",Ridge(alpha=1))
])
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
model_mse=mean_squared_error(y_test,y_pred)
baseline=np.zeros_like(y_test)
baseline_mse=mean_squared_error(y_test,baseline)
oos_r2=1-model_mse/baseline_mse
print("Model MSE",model_mse)
print("Baseline MSE",baseline_mse)
print("OOS R2",oos_r2)
#alpha=1
#OOS R2： -0.0178