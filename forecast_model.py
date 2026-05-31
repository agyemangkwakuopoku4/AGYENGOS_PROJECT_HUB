import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv("data/retail_store_inventory.csv")

# Convert date
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date")

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

# Lag features (VERY important)
data["lag_1"] = data["Units Sold"].shift(1)
data["lag_2"] = data["Units Sold"].shift(2)
data["lag_3"] = data["Units Sold"].shift(3)

# Rolling mean (trend smoothing)
data["rolling_3"] = data["Units Sold"].rolling(window=3).mean()

# Date features
data["day"] = data["Date"].dt.day
data["month"] = data["Date"].dt.month
data["weekday"] = data["Date"].dt.weekday

# Drop missing rows caused by lagging
data = data.dropna()

# -----------------------------
# FEATURES & TARGET
# -----------------------------
features = [
    "Inventory Level",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_3",
    "day",
    "month",
    "weekday"
]

X = data[features]
y = data["Units Sold"]

# Train-test split (time aware)
train_size = int(len(data) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# -----------------------------
# MODEL
# -----------------------------
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_demand():
    latest = X.iloc[-1].values.reshape(1, -1)
    return model.predict(latest)[0]


__all__ = ["predict_demand"]
