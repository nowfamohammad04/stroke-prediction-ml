import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Load dataset
data = pd.read_csv("dataset/stroke_data.csv")

# Drop id column
if "id" in data.columns:
    data = data.drop("id", axis=1)

# Fill missing BMI
data['bmi'] = data['bmi'].fillna(data['bmi'].mean())

# One-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Split features and target
X = data.drop("stroke", axis=1)
y = data["stroke"]

# -------- SCALING --------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------- BALANCING --------
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# Train model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Save files
joblib.dump(model, "models/stroke_model.pkl")
joblib.dump(X.columns.tolist(), "models/features.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model trained successfully with balanced features!")