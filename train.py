import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


data = fetch_california_housing()
X = data.data
y = data.target

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("California_Housing_Price_Prediction")

experiments = [
    {"run_name": "Run 1", "max_depth": 3, "learning_rate": 0.1},
    {"run_name": "Run 2", "max_depth": 5, "learning_rate": 0.05},
    {"run_name": "Run 3", "max_depth": 7, "learning_rate": 0.01},
]

for exp in experiments:
    run_name = exp["run_name"]
    max_depth = exp["max_depth"]
    learning_rate = exp["learning_rate"]

    with mlflow.start_run(run_name=run_name):

        model = GradientBoostingRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)

        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        print(f"{run_name} Results-> RMSE: {rmse:.3f} | MAE: {mae:.3f} | R2: {r2:.3f}")