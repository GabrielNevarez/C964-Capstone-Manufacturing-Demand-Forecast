import pandas as pd

FILE_PATH = "data/raw/Historical Product Demand.csv"
OUTPUT_PATH = "data/cleaned/Historical_Product_Demand_Cleaned.csv"

df = pd.read_csv(FILE_PATH)

# Remove rows without a date
df = df.dropna(subset=["Date"]).copy()

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Strip whitespace
df["Order_Demand"] = df["Order_Demand"].astype(str).str.strip()

# Convert values like (100) to -100
negative_mask = df["Order_Demand"].str.match(r"^\(\d+\)$", na=False)

df.loc[negative_mask, "Order_Demand"] = (
    "-" +
    df.loc[negative_mask, "Order_Demand"]
      .str.replace("(", "", regex=False)
      .str.replace(")", "", regex=False)
)

# Convert demand to numeric
df["Order_Demand"] = pd.to_numeric(
    df["Order_Demand"],
    errors="coerce"
)

# Remove rows that still failed conversion
df = df.dropna(subset=["Order_Demand"]).copy()

# Sort the cleaned data
df = df.sort_values(
    by=["Product_Code", "Warehouse", "Date"]
)

# Save cleaned dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\n--- CLEANING COMPLETE ---")
print("Rows:", len(df))
print("Missing dates:", df["Date"].isna().sum())
print("Missing demand:", df["Order_Demand"].isna().sum())
print("Negative demand rows:", (df["Order_Demand"] < 0).sum())
print("Zero demand rows:", (df["Order_Demand"] == 0).sum())
print("Date range:", df["Date"].min(), "to", df["Date"].max())
print("\nSaved to:", OUTPUT_PATH)