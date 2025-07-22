import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Heart Disease Risk App", layout="centered")
st.title("💓 Heart Disease Risk Assessment")
st.write("Enter patient data to predict risk")

# Load model and scaler
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Input form
age = st.slider("Age", 20, 80)
sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure", 80, 200)
chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [1, 0])
restecg = st.selectbox("Rest ECG (0-2)", [0, 1, 2])
thalach = st.slider("Max Heart Rate Achieved", 70, 210)
exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [1, 0])
oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, step=0.1)
slope = st.selectbox("Slope of ST (0-2)", [0, 1, 2])
ca = st.selectbox("Major Vessels Colored (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1=Normal, 2=Fixed defect, 3=Reversible defect)", [1, 2, 3])

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    if prediction[0] == 1:
        st.error("High Risk of Heart Disease ❌")
    else:
        st.success("Low Risk of Heart Disease ✅") 