import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_purchase_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Purchase Prediction
st.title("Tourism Purchase Prediction Application")
st.write("""
An interactive, real-time prediction dashboard designed for sales teams to identify 
high-potential customers for our Wellness Tourism Package before initiating contact.
""")

st.subheader("Customer Demographics & Profile")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", min_value=15, max_value=65, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Fe Male"])
    marital_status = st.selectbox(
        "Marital Status", ["Married", "Single", "Unmarried", "Divorced"]
    )
    occupation = st.selectbox(
        "Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )

with col2:
    designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    monthly_income = st.number_input(
        "Monthly Income", min_value=0.0, value=25000.0, step=500.0
    )
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    passport = st.selectbox("Has Passport?", ["Yes", "No"])

with col3:
    own_car = st.selectbox("Owns a Car?", ["Yes", "No"])
    num_trips = st.number_input(
        "Number of Annual Trips", min_value=0, max_value=20, value=3
    )
    preferred_star = st.selectbox("Preferred Property Star Rating", [3, 4, 5])

st.subheader("Travel Group Details")
col4, col5 = st.columns(2)
with col4:
    num_visitors = st.number_input(
        "Total Number of Persons Visiting", min_value=1, max_value=10, value=2
    )
with col5:
    num_children = st.number_input(
        "Number of Children Visiting (Age < 5)", min_value=0, max_value=5, value=0
    )

st.subheader("Sales Interaction Details")
col6, col7, col8 = st.columns(3)

with col6:
    contact_type = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    product_pitched = st.selectbox(
        "Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
    )

with col7:
    pitch_duration = st.number_input(
        "Duration of Pitch (Minutes)", min_value=0, max_value=120, value=15
    )
    pitch_satisfaction = st.slider(
        "Pitch Satisfaction Score", min_value=1, max_value=5, value=3
    )

with col8:
    num_followups = st.selectbox(
        "Number of Follow-ups", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=3
    )

# Mapping binary categorical UI selections back to standard numeric format if expected by the pipeline
passport_val = 1 if passport == "Yes" else 0
own_car_val = 1 if own_car == "Yes" else 0

# Assemble input into DataFrame matching the expected schema of your pipeline
input_data = pd.DataFrame(
    [
        {
            "Age": age,
            "TypeofContact": contact_type,
            "CityTier": city_tier,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": num_visitors,
            "PreferredPropertyStar": preferred_star,
            "MaritalStatus": marital_status,
            "NumberOfTrips": num_trips,
            "Passport": passport_val,
            "OwnCar": own_car_val,
            "NumberOfChildrenVisiting": num_children,
            "Designation": designation,
            "MonthlyIncome": monthly_income,
            "PitchSatisfactionScore": pitch_satisfaction,
            "ProductPitched": product_pitched,
            "NumberOfFollowups": num_followups,
            "DurationOfPitch": pitch_duration,
        }
    ]
)

# Predict
if st.button("Predict Purchase Decision"):
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(
            "🎉 **Highly Likely to Purchase!** The model predicts this customer will buy the Wellness Tourism Package."
        )
    else:
        st.info(
            "⚠️ **Unlikely to Purchase.** The model predicts this customer is not interested at this stage."
        )
