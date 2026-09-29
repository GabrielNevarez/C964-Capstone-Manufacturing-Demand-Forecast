import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

INPUT_PATH = "data/cleaned/Weekly_Product_Demand_Features.csv"

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

# Use the oldest 80% for training and newest 20% for testing
# Split data by date
split_date = df["Week"].quantile(0.80)

train_df = df[df["Week"] < split_date].copy()
test_df = df[df["Week"] >= split_date].copy()

print("Split date:", split_date)

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate model accuracy metrics
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

# Print results
print("\n--- LINEAR REGRESSION RESULTS ---")
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

print("\n--- MODEL ACCURACY ---")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2:", round(r2, 4))

print("\n--- SAMPLE PREDICTIONS ---")

results = test_df[
    ["Week", "Product_Code", "Warehouse", "Total_Demand"]
].copy()

results["Predicted_Demand"] = predictions

print(results.head(20))
# Train random forest model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test)

# Calculate model accuracy metrics
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)

# Print random forest results
print("\n--- RANDOM FOREST RESULTS ---")

print("\n--- MODEL ACCURACY ---")
print("MAE:", round(rf_mae, 2))
print("RMSE:", round(rf_rmse, 2))
print("R2:", round(rf_r2, 4))
