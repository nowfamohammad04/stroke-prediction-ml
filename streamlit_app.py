import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("models/stroke_model.pkl", "rb"))

st.title("🧠 Stroke Prediction System")

st.write("Enter patient details:")

# Inputs
age = st.slider("Age", 1, 100, 25)
hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])
heart_disease = st.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1])
glucose = st.number_input("Glucose Level", 50.0, 300.0, 100.0)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

# Prediction
if st.button("Predict"):
    input_data = np.array([[age, hypertension, heart_disease, glucose, bmi]])
    input_df = pd.DataFrame(input_data, columns=features)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High Stroke Risk")
    else:
        st.success("✅ Low Stroke Risk")

    st.write(f"Probability: {probability[0][1]*100:.2f}%")