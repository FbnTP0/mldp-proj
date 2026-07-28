import base64
import joblib
import streamlit as st
import pandas as pd


# Set the background image
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

        div.block-container {{
            background-color: rgba(0, 0, 0, 0.88) !important;
            border-radius: 20px;
            padding: 2rem !important;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.45);
            max-width: 800px;
        }}

        h1, h2, h3, p, label, .stMarkdown {{
            color: #f8f8f2 !important;
        }}

        .stButton > button {{
            background-color: grey !important;
            color: #f8f8f2 !important;
            border: 1px solid #666 !important;
            border-radius: 10px;
            width: 100%;
        }}

        .stButton > button:hover {{
            background-color: #555 !important;
            border-color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# Load trained model
model = joblib.load("insurance_model.pkl")

# Set background
set_background_image("image.jpg")


# App title
st.title("Insurance Charges Prediction")

st.write(
    "Enter your details below to estimate your annual insurance charges."
)


# User inputs
age = st.slider(
    "Age",
    min_value=18,
    max_value=64,
    value=30
)

sex = st.selectbox(
    "Sex",
    options=["female", "male"]
)

height_cm = st.number_input(
    "Height (cm)",
    min_value=100.0,
    max_value=250.0,
    value=170.0,
    step=1.0
)

weight_kg = st.number_input(
    "Weight (kg)",
    min_value=30.0,
    max_value=200.0,
    value=70.0,
    step=1.0
)

# Calculate BMI automatically
bmi = weight_kg / ((height_cm / 100) ** 2)

st.write(f"Calculated BMI: **{bmi:.2f}**")

children = st.slider(
    "Number of Children",
    min_value=0,
    max_value=5,
    value=0
)

# Smoker toggle
smoker_toggle = st.toggle(
    "Smoker?",
    value=False
)

smoker_yes = 1 if smoker_toggle else 0

region = st.selectbox(
    "Region",
    options=[
        "northeast",
        "northwest",
        "southeast",
        "southwest"
    ]
)


# Predict button
if st.button("Predict Insurance Charges"):

    input_encoded = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex_male": [1 if sex == "male" else 0],
        "smoker_yes": [smoker_yes],
        "region_northwest": [1 if region == "northwest" else 0],
        "region_southeast": [1 if region == "southeast" else 0],
        "region_southwest": [1 if region == "southwest" else 0]
    })

    # Use the exact columns that the saved model was trained with
    if hasattr(model, "feature_names_in_"):
        model_columns = list(model.feature_names_in_)
    else:
        model_columns = [
            "age",
            "bmi",
            "children",
            "sex_male",
            "smoker_yes",
            "region_northwest",
            "region_southeast",
            "region_southwest"
        ]

    input_encoded = input_encoded.reindex(
        columns=model_columns,
        fill_value=0
    )

    try:
        prediction = model.predict(input_encoded)[0]

        # Prevent a negative lower range
        margin = max(500.0, prediction * 0.10)
        lower_range = max(0, prediction - margin)
        upper_range = prediction + margin

        st.success(
            f"Predicted Insurance Charges: ${prediction:,.2f}"
        )

        st.info(
            f"Approximate range: "
            f"${lower_range:,.2f} to ${upper_range:,.2f}"
        )

        st.caption(
            "The displayed range is an approximate 10% range and is not "
            "a model-calculated confidence interval."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")