import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("dataset/stroke_data.csv")

# Fix missing BMI values
data['bmi'].fillna(data['bmi'].mean(), inplace=True)

# Convert categorical text columns into numbers
data = pd.get_dummies(data, drop_first=True)

# Separate features and target
X = data.drop("stroke", axis=1)
y = data["stroke"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data size:", X_train.shape)
print("Testing data size:", X_test.shape)