import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# Load dataset
df = pd.read_csv("train.csv")

# Remove rows with missing target
df = df.dropna(subset=["SalePrice"])

# Fill missing values
df = df.fillna(0)

# Separate target
y = df["SalePrice"]
X = df.drop("SalePrice", axis=1)

# Convert categorical columns automatically
X = pd.get_dummies(X)

# Save column names
model_columns = X.columns.tolist()
joblib.dump(model_columns, "columns.pkl")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Train model
model = GradientBoostingRegressor(n_estimators=100)

model.fit(X_scaled, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model Saved Successfully")