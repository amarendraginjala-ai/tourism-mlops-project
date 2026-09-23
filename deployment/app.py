
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Tourism Package Purchase Prediction",
    page_icon="✈️",
    layout="centered"
)

st.title("Tourism Package Purchase Prediction")

st.write(
    "Enter the customer details below to predict whether "
    "the customer is likely to purchase the tourism package."
)


# ---------------------------------------------------------
# Load the trained model from Hugging Face Model Hub
# ---------------------------------------------------------

@st.cache_resource
def load_model():

    # Define the Hugging Face Model Hub repository
    model_repository = "Amarendraa/Tourism-Package-Purchase-RandomForest"

    # Download the trained Random Forest model
    model_path = hf_hub_download(
        repo_id=model_repository,
        filename="random_forest_tourism_model.joblib"
    )

    # Load the saved model
    model = joblib.load(model_path)

    return model


model = load_model()


# ---------------------------------------------------------
# Collect customer details
# ---------------------------------------------------------

st.subheader("Customer Details")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

type_of_contact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

city_tier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Free Lancer", "Small Business", "Large Business"]
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

number_of_person_visiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=20,
    value=2
)

preferred_property_star = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Divorced", "Married", "Unmarried"]
)

number_of_trips = st.number_input(
    "Number of Trips",
    min_value=1,
    max_value=30,
    value=3
)

passport = st.selectbox(
    "Passport",
    [0, 1]
)

own_car = st.selectbox(
    "Own Car",
    [0, 1]
)

number_of_children_visiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=10,
    value=0
)

designation = st.selectbox(
    "Designation",
    ["Manager", "Executive", "Senior Manager", "AVP", "VP"]
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=0.0,
    value=20000.0
)


# ---------------------------------------------------------
# Customer interaction details
# ---------------------------------------------------------

st.subheader("Customer Interaction Details")

pitch_satisfaction_score = st.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5]
)

product_pitched = st.selectbox(
    "Product Pitched",
    ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"]
)

number_of_followups = st.number_input(
    "Number of Follow-ups",
    min_value=1.0,
    max_value=10.0,
    value=4.0
)

duration_of_pitch = st.number_input(
    "Duration of Pitch",
    min_value=1.0,
    max_value=60.0,
    value=10.0
)


# ---------------------------------------------------------
# Make prediction
# ---------------------------------------------------------

if st.button("Predict Package Purchase"):

    # Create a DataFrame containing the user inputs
    input_data = pd.DataFrame({
        "Age": [age],
        "TypeofContact": [type_of_contact],
        "CityTier": [city_tier],
        "Occupation": [occupation],
        "Gender": [gender],
        "NumberOfPersonVisiting": [number_of_person_visiting],
        "PreferredPropertyStar": [preferred_property_star],
        "MaritalStatus": [marital_status],
        "NumberOfTrips": [number_of_trips],
        "Passport": [passport],
        "OwnCar": [own_car],
        "NumberOfChildrenVisiting": [number_of_children_visiting],
        "Designation": [designation],
        "MonthlyIncome": [monthly_income],
        "PitchSatisfactionScore": [pitch_satisfaction_score],
        "ProductPitched": [product_pitched],
        "NumberOfFollowups": [number_of_followups],
        "DurationOfPitch": [duration_of_pitch]
    })


    # -----------------------------------------------------
    # Apply the same preprocessing used during model training
    # -----------------------------------------------------

    # Encode Type of Contact
    input_data["TypeofContact"] = input_data["TypeofContact"].map({
        "Self Enquiry": 0,
        "Company Invited": 1
    })

    # Encode Gender
    input_data["Gender"] = input_data["Gender"].map({
        "Female": 0,
        "Male": 1
    })

    # Apply one-hot encoding to categorical variables
    input_data = pd.get_dummies(
        input_data,
        columns=[
            "Occupation",
            "ProductPitched",
            "MaritalStatus",
            "Designation"
        ],
        drop_first=True,
        dtype=int
    )


    # -----------------------------------------------------
    # Match the exact training dataset feature structure
    # -----------------------------------------------------

    # Load only the column names from the training dataset
    train_url = (
        "https://huggingface.co/datasets/Amarendraa/Tourism/"
        "resolve/main/train.csv"
    )

    training_columns = pd.read_csv(
        train_url,
        nrows=0
    ).columns.tolist()

    # Remove the target variable because the model only receives features
    expected_features = [
        column for column in training_columns
        if column != "ProdTaken"
    ]

    # Arrange the input columns in exactly the same order as training
    # Missing dummy variables are automatically filled with zero
    input_data = input_data.reindex(
        columns=expected_features,
        fill_value=0
    )


    # -----------------------------------------------------
    # Generate prediction
    # -----------------------------------------------------

    prediction = model.predict(input_data)[0]

    # Calculate the probability of package purchase
    probability = model.predict_proba(input_data)[0][1]


    # -----------------------------------------------------
    # Display prediction
    # -----------------------------------------------------

    if prediction == 1:
        st.success(
            "Prediction: Customer is likely to purchase the package."
        )
    else:
        st.info(
            "Prediction: Customer is unlikely to purchase the package."
        )

    st.write(
        f"Estimated purchase probability: {probability:.2%}"
    )
