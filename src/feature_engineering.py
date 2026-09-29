import pandas as pd

INPUT_PATH = "data/cleaned/Historical_Product_Demand_Cleaned.csv"
OUTPUT_PATH = "data/cleaned/Weekly_Product_Demand_Features.csv"

# Load cleaned data
df = pd.read_csv(INPUT_PATH)

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Create weekly date bucket
df["Week"] = df["Date"].dt.to_period("W").apply(lambda x: x.start_time)

# Aggregate weekly demand
weekly_df = (
    df.groupby(
        ["Product_Code", "Warehouse", "Product_Category", "Week"],
        as_index=False
    )["Order_Demand"]
    .sum()
    .rename(columns={"Order_Demand": "Total_Demand"})
)

# Fill missing calendar weeks
completed_groups = []

for (product, warehouse), group in weekly_df.groupby(
    ["Product_Code", "Warehouse"]
):

    group = group.sort_values("Week").copy()

    # Create full weekly range
    full_weeks = pd.date_range(
        start=group["Week"].min(),
        end=group["Week"].max(),
        freq="W-MON"
    )

    # Insert missing weeks
    group = group.set_index("Week")
    group = group.reindex(full_weeks)
    group.index.name = "Week"

    # Restore product information
    group["Product_Code"] = product
    group["Warehouse"] = warehouse

    category = (
        weekly_df.loc[
            weekly_df["Product_Code"] == product,
            "Product_Category"
        ]
        .dropna()
        .iloc[0]
    )

    group["Product_Category"] = category

    # Missing weeks are treated as zero demand
    group["Total_Demand"] = group["Total_Demand"].fillna(0)

    group = group.reset_index()

    completed_groups.append(group)

# Combine completed timelines
weekly_df = pd.concat(
    completed_groups,
    ignore_index=True
)

# Sort data
weekly_df = weekly_df.sort_values(
    by=["Product_Code", "Warehouse", "Week"]
).reset_index(drop=True)

# Create previous week features
weekly_df["Prev_Week1"] = weekly_df.groupby(
    ["Product_Code", "Warehouse"]
)["Total_Demand"].shift(1)

weekly_df["Prev_Week2"] = weekly_df.groupby(
    ["Product_Code", "Warehouse"]
)["Total_Demand"].shift(2)

weekly_df["Prev_Week3"] = weekly_df.groupby(
    ["Product_Code", "Warehouse"]
)["Total_Demand"].shift(3)

weekly_df["Prev_Week4"] = weekly_df.groupby(
    ["Product_Code", "Warehouse"]
)["Total_Demand"].shift(4)

# Create 4 week rolling average
weekly_df["Rolling_Mean_4"] = weekly_df.groupby(
    ["Product_Code", "Warehouse"]
)["Total_Demand"].transform(
    lambda x: x.shift(1).rolling(window=4).mean()
)

# Add time features
weekly_df["Year"] = weekly_df["Week"].dt.year
weekly_df["Month"] = weekly_df["Week"].dt.month
weekly_df["Week_of_Year"] = (
    weekly_df["Week"]
    .dt.isocalendar()
    .week
    .astype(int)
)

# Remove rows without enough history
model_df = weekly_df.dropna(
    subset=[
        "Prev_Week1",
        "Prev_Week2",
        "Prev_Week3",
        "Prev_Week4",
        "Rolling_Mean_4"
    ]
).copy()

# Save model ready data
model_df.to_csv(
    OUTPUT_PATH,
    index=False
)

# Print summary
print("\n--- FEATURE ENGINEERING COMPLETE ---")
print("Weekly rows after filling missing weeks:", len(weekly_df))
print("Model-ready rows:", len(model_df))
print("Inserted zero-demand weeks:", (weekly_df["Total_Demand"] == 0).sum())

print("\n--- COLUMNS ---")
print(model_df.columns.tolist())

print("\n--- FIRST 15 ROWS ---")
print(model_df.head(15))

print("\n--- DATE RANGE ---")
print("Earliest week:", model_df["Week"].min())
print("Latest week:", model_df["Week"].max())

print("\nSaved to:", OUTPUT_PATH)