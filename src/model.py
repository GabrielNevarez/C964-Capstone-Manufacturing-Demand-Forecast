import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


INPUT_PATH = "data/cleaned/Weekly_Product_Demand_Features.csv"
RESULTS_PATH = "data/cleaned/Model_Comparison.csv"
MODEL_PATH = "models/linear_regression_model.joblib"


# Load model-ready data
df = pd.read_csv(INPUT_PATH)


# Convert Week to datetime
df["Week"] = pd.to_datetime(df["Week"])


# Sort by date
df = df.sort_values("Week").reset_index(drop=True)


# Select model features
features = [
    "Prev_Week1",
    "Prev_Week2",
    "Prev_Week3",
    "Prev_Week4",
    "Rolling_Mean_4",
    "Month",
    "Week_of_Year"
]

target = "Total_Demand"


# Split data by date
split_date = df["Week"].quantile(0.80)

train_df = df[df["Week"] < split_date].copy()
test_df = df[df["Week"] >= split_date].copy()

print("Split date:", split_date)


# Create training data
X_train = train_df[features]
y_train = train_df[target]


# Create testing data
X_test = test_df[features]
y_test = test_df[target]


# Train linear regression model
linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)


# Make linear regression predictions
linear_predictions = linear_model.predict(X_test)


# Calculate linear regression accuracy
linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_predictions
    )
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)


# Print linear regression results
print("\n--- LINEAR REGRESSION RESULTS ---")

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

print("\n--- MODEL ACCURACY ---")

print("MAE:", round(linear_mae, 2))
print("RMSE:", round(linear_rmse, 2))
print("R2:", round(linear_r2, 4))


# Train random forest model
random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

random_forest_model.fit(
    X_train,
    y_train
)


# Make random forest predictions
random_forest_predictions = random_forest_model.predict(
    X_test
)


# Calculate random forest accuracy
random_forest_mae = mean_absolute_error(
    y_test,
    random_forest_predictions
)

random_forest_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        random_forest_predictions
    )
)

random_forest_r2 = r2_score(
    y_test,
    random_forest_predictions
)


# Print random forest results
print("\n--- RANDOM FOREST RESULTS ---")

print("MAE:", round(random_forest_mae, 2))
print("RMSE:", round(random_forest_rmse, 2))
print("R2:", round(random_forest_r2, 4))


# Create model comparison
model_results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"
    ],
    "MAE": [
        linear_mae,
        random_forest_mae
    ],
    "RMSE": [
        linear_rmse,
        random_forest_rmse
    ],
    "R2": [
        linear_r2,
        random_forest_r2
    ]
})


# Save model comparison
model_results.to_csv(
    RESULTS_PATH,
    index=False
)


# Save linear regression model
joblib.dump(
    linear_model,
    MODEL_PATH
)


# Print model comparison
print("\n--- MODEL COMPARISON ---")

print(model_results)


# Print saved files
print(
    "\nSaved model comparison to:",
    RESULTS_PATH
)

print(
    "Saved linear regression model to:",
    MODEL_PATH
)