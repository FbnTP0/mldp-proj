import joblib
import streamlit as st
import numpy as np
import pandas as pd

## Load trained model
model = joblib.load("insurance_model.pkl")

## App title
st.title("Insurance Charges Prediction")
st.write("Enter your details below to estimate your insurance charges.")

## User inputs
age = st.slider("Age", min_value=18, max_value=64, value=30)
bmi = st.number_input("BMI", min_value=15.0, max_value=55.0, value=25.0)
children = st.slider("Number of Children", min_value=0, max_value=5, value=0)
smoker = st.selectbox("Smoker", ["yes", "no"])

## Predict button
if st.button("Predict Insurance Charges"):
    ## Build a single-row dataframe matching original columns
    input_df = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
    })

    ## Apply same one-hot encoding used in training
    input_encoded = pd.get_dummies(input_df, columns=["smoker"], drop_first=True)

    ## Align columns exactly to what the model was trained on
    ## (replace this list with your actual X.columns from training)
    model_columns = ['age', 'bmi', 'children', 'smoker_yes']
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    ## Predict
    prediction = model.predict(input_encoded)[0]
    st.success(f"Predicted Insurance Charges: ${prediction:,.2f}")