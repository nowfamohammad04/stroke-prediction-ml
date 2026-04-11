import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# -------- LOAD --------
model = joblib.load("models/stroke_model.pkl")
features = joblib.load("models/features.pkl")
features = list(features)
scaler = joblib.load("models/scaler.pkl")

# -------- PAGE CONFIG --------
st.set_page_config(page_title="Stroke Prediction", layout="centered")

# -------- CUSTOM CSS --------
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}
.stButton>button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# -------- HEADER --------
st.markdown("<h1 style='text-align:center; color:#6366f1;'>🧠 Stroke Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered health risk analysis</p>", unsafe_allow_html=True)

st.markdown("---")

# -------- INPUT SECTION --------
st.subheader("📝 Enter Your Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 1, 100, 25)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    glucose = st.number_input("Glucose Level", 50.0, 300.0, 100.0)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children"])

hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])

st.markdown("---")

# -------- BUTTON --------
predict = st.button("🚀 Predict Stroke Risk")

risk_prob = None

# -------- PREDICTION --------
if predict:
    try:
        input_dict = {col: 0 for col in features}

        input_dict["age"] = age
        input_dict["hypertension"] = hypertension
        input_dict["heart_disease"] = heart_disease
        input_dict["avg_glucose_level"] = glucose
        input_dict["bmi"] = bmi

        # One-hot encoding
        if f"gender_{gender}" in features:
            input_dict[f"gender_{gender}"] = 1

        if f"ever_married_{ever_married}" in features:
            input_dict[f"ever_married_{ever_married}"] = 1

        if f"work_type_{work_type}" in features:
            input_dict[f"work_type_{work_type}"] = 1

        if f"Residence_type_{residence_type}" in features:
            input_dict[f"Residence_type_{residence_type}"] = 1

        if f"smoking_status_{smoking_status}" in features:
            input_dict[f"smoking_status_{smoking_status}"] = 1

        # DataFrame
        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=features, fill_value=0)

        # Scaling
        input_scaled = scaler.transform(input_df.astype(float))

        # Prediction
        probability = model.predict_proba(input_scaled)
        risk_prob = float(probability[0][1] * 100)

    except Exception as e:
        st.error(f"Error: {e}")

# -------- OUTPUT --------
if risk_prob is not None:

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    st.metric(label="Stroke Risk", value=f"{risk_prob:.2f}%")

    # progress bar
    st.progress(int(risk_prob))

    # chart
    fig, ax = plt.subplots()
    ax.pie([100 - risk_prob, risk_prob],
           labels=["Safe", "Risk"],
           autopct='%1.1f%%')
    st.pyplot(fig)

    # risk level
    if risk_prob < 35:
        st.success("🟢 Low Risk")
    elif risk_prob < 80:
        st.warning("🟡 Moderate Risk")
    else:
        st.error("🔴 High Risk")

    # recommendations
    st.markdown("### 💡 Personalized Recommendations")

    recommendations = []

    if bmi > 25:
        recommendations.append("Maintain a healthy BMI")

    if glucose > 140:
        recommendations.append("Control blood sugar levels")

    if hypertension == 1:
        recommendations.append("Monitor blood pressure")

    if heart_disease == 1:
        recommendations.append("Regular cardiac checkups")

    if smoking_status == "smokes":
        recommendations.append("Quit smoking")
    elif smoking_status == "formerly smoked":
        recommendations.append("Continue avoiding smoking")

    if age > 50:
        recommendations.append("Regular health screening recommended")

    if len(recommendations) == 0:
        recommendations.append("Maintain your healthy lifestyle")

    for rec in recommendations:
        st.write(f"✔ {rec}")

# -------- FOOTER --------
st.markdown("---")
st.info("⚠️ This is an AI-based prediction and not a medical diagnosis.")