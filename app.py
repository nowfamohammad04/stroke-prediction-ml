import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = joblib.load("models/stroke_model.pkl")
features = joblib.load("models/features.pkl")

st.set_page_config(page_title="Stroke Prediction", layout="wide")

# -------- STYLING --------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.header {
    padding: 20px;
    border-radius: 12px;
    background: linear-gradient(90deg, #141e30, #243b55);
    color: white;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background: #1e1e1e;
    margin-bottom: 20px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.3);
}
.result {
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
    color: white;
    text-align: center;
}
.section-title {
    font-size: 20px;
    font-weight: bold;
    color: #00c6ff;
}
</style>
""", unsafe_allow_html=True)

# -------- HEADER --------
st.markdown("""
<div class="header">
<h1>🧠 Stroke Risk Prediction Dashboard</h1>
<p>AI-powered healthcare risk analysis system</p>
</div>
""", unsafe_allow_html=True)

st.markdown("")

# -------- LAYOUT --------
col1, col2 = st.columns(2)

# LEFT SIDE
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🧍 Personal Info</p>', unsafe_allow_html=True)

    age = st.slider("Age", 1, 100, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">❤️ Health Conditions</p>', unsafe_allow_html=True)

    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])

    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT SIDE
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🧪 Health Metrics</p>', unsafe_allow_html=True)

    glucose = st.number_input("Glucose Level", 50.0, 300.0, 100.0)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🧠 Lifestyle</p>', unsafe_allow_html=True)

    work_type = st.selectbox(
        "Work Type",
        ["Private", "Self-employed", "Govt_job", "children"]
    )

    Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])

    smoking_status = st.selectbox(
        "Smoking Status",
        ["formerly smoked", "never smoked", "smokes"]
    )

    st.markdown('</div>', unsafe_allow_html=True)

# BUTTON
predict = st.button("🚀 Analyze Risk")

# -------- PREDICTION --------
if predict:

    input_dict = {
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "avg_glucose_level": glucose,
        "bmi": bmi,

        "gender_Male": 1 if gender == "Male" else 0,
        "ever_married_Yes": 1 if ever_married == "Yes" else 0,
        "work_type_Private": 1 if work_type == "Private" else 0,
        "work_type_Self-employed": 1 if work_type == "Self-employed" else 0,
        "work_type_children": 1 if work_type == "children" else 0,
        "Residence_type_Urban": 1 if Residence_type == "Urban" else 0,
        "smoking_status_formerly smoked": 1 if smoking_status == "formerly smoked" else 0,
        "smoking_status_never smoked": 1 if smoking_status == "never smoked" else 0,
        "smoking_status_smokes": 1 if smoking_status == "smokes" else 0,
    }

    input_df = pd.DataFrame([input_dict])

    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[features]

    probability = model.predict_proba(input_df)
    risk_prob = probability[0][1] * 100

    if risk_prob < 1:
        risk_prob = 1.0

    # RESULT
    st.markdown(f"""
    <div class="result">
        <h2>📊 Stroke Risk: {risk_prob:.2f}%</h2>
    </div>
    """, unsafe_allow_html=True)

    # PROGRESS
    st.progress(int(risk_prob))

    # CHART
    fig, ax = plt.subplots()
    ax.pie([100-risk_prob, risk_prob],
           labels=["Safe", "Risk"],
           autopct='%1.1f%%')
    st.pyplot(fig)

    # -------- FEEDBACK + RECOMMENDATIONS --------

    if risk_prob < 30:
        st.success("🟢 Low Stroke Risk")
        st.write("### 🟢 Interpretation")
        st.write("Your current health indicators suggest a low probability of stroke.")

        st.write("### 💡 Recommendations")
        st.write("""
        - Maintain a balanced diet
        - Exercise regularly (30 mins daily)
        - Avoid smoking and alcohol
        - Regular health checkups
        """)

    elif risk_prob < 60:
        st.warning("🟡 Moderate Stroke Risk")
        st.write("### 🟡 Interpretation")
        st.write("Some risk factors are present. Lifestyle changes are recommended.")

        st.write("### 💡 Recommendations")
        st.write("""
        - Reduce sugar and salt intake
        - Increase physical activity
        - Monitor blood pressure regularly
        - Maintain healthy BMI
        - Avoid smoking
        """)

    else:
        st.error("🔴 High Stroke Risk")
        st.write("### 🔴 Interpretation")
        st.write("High risk detected. Immediate medical attention is advised.")

        st.write("### 💡 Recommendations")
        st.write("""
        - Consult a doctor immediately
        - Control blood pressure and glucose
        - Quit smoking immediately
        - Follow prescribed medications
        - Regular monitoring and checkups
        """)

# FOOTER
st.info("⚠️ This is an AI-based prediction and not a medical diagnosis.")