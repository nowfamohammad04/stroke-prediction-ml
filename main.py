import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
data = pd.read_csv("dataset/stroke_data.csv")

# Drop id safely
if "id" in data.columns:
    data = data.drop("id", axis=1)

# Fill missing BMI
data['bmi'].fillna(data['bmi'].mean(), inplace=True)

# Encode categorical features
data = pd.get_dummies(data, drop_first=True)

# Split
X = data.drop("stroke", axis=1)
y = data["stroke"]

# Balance data
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# Train model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Save model + feature list
joblib.dump(model, "models/stroke_model.pkl")
joblib.dump(X.columns.tolist(), "models/features.pkl")

print("✅ Model trained successfully!")