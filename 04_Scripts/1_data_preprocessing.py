# - - - Preprocessing data - - - #

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 1. Load raw datasets
train = pd.read_csv("R:/Projects/1_Data_Science & ML_Projects/04_Sales and Demand Forecasting for Retail Chains or Predictive Sales/02_Data/Raw_dataset/train.csv", parse_dates=["Date"])
features = pd.read_csv("R:/Projects/1_Data_Science & ML_Projects/04_Sales and Demand Forecasting for Retail Chains or Predictive Sales/02_Data/Raw_dataset/features.csv", parse_dates=["Date"])
stores = pd.read_csv("R:/Projects/1_Data_Science & ML_Projects/04_Sales and Demand Forecasting for Retail Chains or Predictive Sales/02_Data/Raw_dataset/stores.csv")

# 2. Merge datasets
merged = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
merged = merged.merge(stores, on="Store", how="left")

# 3. Handle missing values
numeric_cols = merged.select_dtypes(include=["float64", "int64"]).columns
merged[numeric_cols] = merged[numeric_cols].fillna(0)

categorical_cols = merged.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    merged[col].fillna("Unknown", inplace=True)

# 4. Create date features
merged["Year"] = merged["Date"].dt.year
merged["Month"] = merged["Date"].dt.month
merged["Week"] = merged["Date"].dt.isocalendar().week.astype(int)
merged["DayOfWeek"] = merged["Date"].dt.dayofweek
merged["Day"] = merged["Date"].dt.day

# 5. Aggregate weekly sales by Store and Date
weekly_sales = merged.groupby(["Store", "Date"])["Weekly_Sales"].sum().reset_index()

# 6. Sort by date
weekly_sales = weekly_sales.sort_values("Date")

# 7. Split into train/test (80/20)
train_data, test_data = train_test_split(weekly_sales, test_size=0.2, shuffle=False)

# 8. Save processed data
merged.to_csv("R:/Projects/1_Data_Science & ML_Projects/04_Sales and Demand Forecasting for Retail Chains or Predictive Sales/02_Data/Processed_dataset/merged_data.csv", index=False)
train_data.to_csv("R:/Projects/1_Data_Science & ML_Projects/04_Sales and Demand Forecasting for Retail Chains or Predictive Sales/02_Data/Processed_dataset/train_data.csv", index=False)
test_data.to_csv("R:/Projects/1_Data_Science & ML_Projects/04_Sales and Demand Forecasting for Retail Chains or Predictive Sales/02_Data/Processed_dataset/test_data.csv", index=False)

print("Data preprocessing complete. Files saved in Processed_data folders.")
