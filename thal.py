import streamlit as st
import numpy as np
import joblib

# Load models
stage1_model = joblib.load("Ensemble_Stage_1_(Binary).pkl")
stage2_model = joblib.load("Ensemble_Stage_2_(Multi-Class).pkl")

st.title("🩸 Thalassemia Risk Prediction")

# Input fields
Hb = st.number_input("Hemoglobin (Hb):", step=0.1)
RBC = st.number_input("Red Blood Cell Count (RBC):", step=0.1)
MCV = st.number_input("Mean Corpuscular Volume (MCV):", step=0.1)
MCH = st.number_input("Mean Corpuscular Hemoglobin (MCH):", step=0.1)
MCHC = st.number_input("MCH Concentration (MCHC):", step=0.1)
RDW = st.number_input("Red Cell Distribution Width (RDW):", step=0.1)
Gender = st.selectbox("Gender:", options=["Female (0)", "Male (1)"])
gender_val = 0 if Gender.startswith("Female") else 1

if st.button("🔍 Predict"):
    input_data = np.array([Hb, RBC, MCV, MCH, MCHC, RDW, gender_val]).reshape(1, -1)
    
    stage1_pred = stage1_model.predict(input_data)[0]
    if stage1_pred == 0:
        result = "Normal"
    else:
        stage2_pred = stage2_model.predict(input_data)[0]
        if stage2_pred == 0:
            result = "Silent Carrier"
        elif stage2_pred == 1:
            result = "Alpha Trait"
        else:
            result = "Unknown Type"
    
    st.success(f"🧪 Prediction: {result}")
