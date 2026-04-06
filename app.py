import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# -------- LOAD --------
model = joblib.load("models/stroke_model.pkl")
features = joblib.load("models/features.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="Stroke Prediction", layout="centered")

st.title("🧠 Stroke Prediction System")
st.write("Enter your health details to predict stroke risk")

# -------- INPUTS --------
age = st.slider("Age", 1, 100, 25)
gender = st.selectbox("Gender", ["Male", "Female"])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])

hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])

glucose = st.number_input("Glucose Level", 50.0, 300.0, 100.0)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children"])
Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes"])

# -------- BUTTON --------
predict = st.button("🚀 Predict Stroke Risk")

# -------- SAFE DEFAULT --------
risk_prob = None  # prevents NameError ALWAYS

# -------- PREDICTION --------
if predict:

    input_dict = {col: 0 for col in features}

    # numeric
    input_dict["age"] = age
    input_dict["hypertension"] = hypertension
    input_dict["heart_disease"] = heart_disease
    input_dict["avg_glucose_level"] = glucose
    input_dict["bmi"] = bmi

    # categorical (safe)
    if f"gender_{gender}" in features:
        input_dict[f"gender_{gender}"] = 1

    if f"ever_married_{ever_married}" in features:
        input_dict[f"ever_married_{ever_married}"] = 1

    if f"work_type_{work_type}" in features:
        input_dict[f"work_type_{work_type}"] = 1

    if f"Residence_type_{Residence_type}" in features:
        input_dict[f"Residence_type_{Residence_type}"] = 1

    if f"smoking_status_{smoking_status}" in features:
        input_dict[f"smoking_status_{smoking_status}"] = 1

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=features, fill_value=0)

    input_scaled = scaler.transform(input_df)

    probability = model.predict_proba(input_scaled)
    risk_prob = float(probability[0][1] * 100)

# -------- OUTPUT (SAFE) --------
if risk_prob is not None:

    st.subheader(f"📊 Stroke Risk: {risk_prob:.2f}%")

    fig, ax = plt.subplots()
    ax.pie([100 - risk_prob, risk_prob],
           labels=["Safe", "Risk"],
           autopct='%1.1f%%')
    st.pyplot(fig)

    # -------- FEEDBACK --------
    if risk_prob < 30:
        st.success("🟢 Low Stroke Risk")
        st.write("### 💡 Recommendations")
        st.write("""
        - Maintain a balanced diet  
        - Exercise regularly  
        - Avoid smoking  
        """)

    elif risk_prob < 60:
        st.warning("🟡 Moderate Stroke Risk")
        st.write("### 💡 Recommendations")
        st.write("""
        - Reduce sugar and salt  
        - Monitor BP  
        - Stay active  
        """)

    else:
        st.error("🔴 High Stroke Risk")
        st.write("### 💡 Recommendations")
        st.write("""
        - Consult a doctor  
        - Control BP & glucose  
        - Quit smoking  
        """)

# -------- FOOTER --------
st.info("⚠️ This is an AI-based prediction and not a medical diagnosis.")