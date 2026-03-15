import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("dataset/stroke_data.csv")

# Handle missing BMI
data['bmi'].fillna(data['bmi'].mean(), inplace=True)

# Encode categorical columns
data = pd.get_dummies(data, drop_first=True)

# Split features and target
X = data.drop("stroke", axis=1)
y = data["stroke"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

import joblib

joblib.dump(model, "models/stroke_model.pkl")

print("Model saved successfully")