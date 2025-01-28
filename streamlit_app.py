
import streamlit as st
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load('svm_best_model (2).pkl')
scaler = joblib.load('scaler (2).pkl')  # Load the saved scaler

# Streamlit app
st.title("Purchase Prediction Using SVM")
st.write("This app predicts whether a user will purchase a product based on their details.")

# Input fields
gender = st.radio("Select Gender", ("Male", "Female"))
age = st.number_input("Enter Age", min_value=0, max_value=100, value=25)
estimated_salary = st.number_input("Enter Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)

# Preprocess input
gender_numeric = 1 if gender == "Male" else 0
user_input = np.array([[gender_numeric, age, estimated_salary]])
user_input_scaled = scaler.transform(user_input)

# Predict and display results
if st.button("Predict"):
    prediction = model.predict(user_input_scaled)
    prediction_label = "Purchased" if prediction[0] == 1 else "Not Purchased"
    st.write(f"The predicted outcome is: **{prediction_label}**")

