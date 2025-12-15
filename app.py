import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

from keras.models import load_model

model = load_model("churn_model.keras", compile=False)
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

scaler = joblib.load("scaler.pkl")

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn")


def yes_no(label):
    return 1 if st.selectbox(label, ["No", "Yes"]) == "Yes" else 0

SeniorCitizen = yes_no("Senior Citizen")
Partner = yes_no("Partner")
Dependents = yes_no("Dependents")

gender = st.selectbox("Gender", ["Female", "Male"])
gender_Male = 1 if gender == "Male" else 0

tenure = st.number_input("Tenure (months)", 0, 72, 12)

PhoneService = yes_no("Phone Service")

MultipleLines = st.selectbox(
    "Multiple Lines",
    ["No phone service", "No", "Yes"]
)
MultipleLines_NoPhone = 1 if MultipleLines == "No phone service" else 0
MultipleLines_Yes = 1 if MultipleLines == "Yes" else 0

InternetService = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)
InternetService_Fiber = 1 if InternetService == "Fiber optic" else 0
InternetService_No = 1 if InternetService == "No" else 0

OnlineSecurity = yes_no("Online Security")
OnlineBackup = yes_no("Online Backup")
DeviceProtection = yes_no("Device Protection")
TechSupport = yes_no("Tech Support")
StreamingTV = yes_no("Streaming TV")
StreamingMovies = yes_no("Streaming Movies")


Contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)
Contract_OneYear = 1 if Contract == "One year" else 0
Contract_TwoYear = 1 if Contract == "Two year" else 0

PaperlessBilling = yes_no("Paperless Billing")

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Credit card (automatic)"
    ]
)
Payment_Electronic = 1 if PaymentMethod == "Electronic check" else 0
Payment_Mailed = 1 if PaymentMethod == "Mailed check" else 0
Payment_CreditCard = 1 if PaymentMethod == "Credit card (automatic)" else 0

MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 800.0)


customer = np.array([[
    SeniorCitizen,
    Partner,
    Dependents,
    tenure,
    PhoneService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    PaperlessBilling,
    MonthlyCharges,
    TotalCharges,
    MultipleLines_NoPhone,
    MultipleLines_Yes,
    InternetService_Fiber,
    InternetService_No,
    Contract_OneYear,
    Contract_TwoYear,
    Payment_CreditCard,
    Payment_Electronic,
    Payment_Mailed,
    gender_Male
]])

if st.button("Predict Churn"):
    customer_scaled = scaler.transform(customer)
    prob = model.predict(customer_scaled)[0][0]

    st.write(f"### Churn Probability: {prob:.2f}")

    if prob > 0.5:
        st.error("Customer is likely to churn ❌")
    else:
        st.success("Customer is likely to stay ✅")
