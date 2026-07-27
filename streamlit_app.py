import base64
import joblib
import streamlit as st
import numpy as np
import pandas as pd

def set_background_image(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

## Load trained model
model = joblib.load("insurance_model.pkl")

set_background_image("image.jpg")

## App title
st.title("Insurance Charges Prediction")
st.write("Enter your details below to estimate your insurance charges.")

## User inputs
age = st.slider("Age", min_value=18, max_value=64, value=30)
height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
bmi = weight_kg / ((height_cm / 100) ** 2)
children = st.slider("Number of Children", min_value=0, max_value=5, value=0)
smoker_toggle = st.checkbox("Smoker?", value=False)
smoker = "yes" if smoker_toggle else "no"

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


    model_columns = ['age', 'bmi', 'children', 'smoker_yes']
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    ## Predict
    prediction = model.predict(input_encoded)[0]
    st.success(f"Predicted Insurance Charges: ${prediction:,.2f}")