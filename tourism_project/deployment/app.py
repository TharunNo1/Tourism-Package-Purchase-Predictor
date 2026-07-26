import os
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(
    page_title="Wellness Tourism Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Gradients, Clean Borders, Card Layouts)
st.markdown(
    """
    <style>
    /* Metric Cards Styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Global Container Adjustments */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0px 0px;
        padding: 10px 16px;
        font-weight: 600;
    }
    
    /* Card Container Wrapper */
    div[data-testid="stForm"] {
        border: 1px solid #E0E0E0;
        border-radius: 12px;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load and cache the model
@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(__file__), "best_tourism_purchase_prediction_model_v1.joblib"
    )
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Failed to load model file. Ensure model file is in directory. Error: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1530789253388-582c481c54b0?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        caption="Wellness Tourism Intelligence",
        use_container_width=True,
    )
    st.title("Sales Advisor Assistant")
    st.markdown(
        """
        Use this predictor before sales calls to gauge lead propensity and customize your sales approach.
        
        ---
        **Quick Tips:**
        - Ensure accurate **Monthly Income**.
        - Ensure **Pitch Duration** reflects real call time.
        """
    )
    st.divider()
    st.caption("v1.0.0 • Sales Intelligence System")

# Main Header
st.title("✈️ Tourism Purchase Prediction")
st.markdown("##### Pre-evaluate customer lead propensity for the **Wellness Tourism Package**")
st.write("---")

# Form to be submitted for prediction
with st.form("prediction_form"):
    tab1, tab2, tab3 = st.tabs([
        "👤 Customer Demographics", 
        "🧳 Travel & Lifestyle", 
        "📞 Sales Interaction Details"
    ])

    # Tab 1: Customer Profile
    with tab1:
        st.subheader("Demographics & Income")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", min_value=15, max_value=65, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
        
        with c2:
            occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
            designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        
        with c3:
            monthly_income = st.number_input("Monthly Income ($)", min_value=0.0, value=25000.0, step=500.0)
            city_tier = st.select_slider("City Tier", options=[1, 2, 3], value=1)

    # Tab 2: Travel Profile
    with tab2:
        st.subheader("Travel Habits & Assets")
        t1, t2, t3 = st.columns(3)
        with t1:
            num_trips = st.number_input("Annual Trips Taken", min_value=0, max_value=20, value=3)
            preferred_star = st.radio("Preferred Hotel Rating", [1, 2, 3, 4, 5], horizontal=True)
            
        with t2:
            num_visitors = st.number_input("Total Visitors in Group", min_value=1, max_value=10, value=2)
            num_children = st.number_input("Children Visiting (< 5 yrs)", min_value=0, max_value=5, value=0)

        with t3:
            passport = st.segmented_control("Has Passport?", ["No", "Yes"], default="Yes", key="passport_segment")
            own_car = st.segmented_control("Owns a Car?", ["No", "Yes"], default="Yes", key="car_segment")

    # Tab 3: Sales Pitch Info
    with tab3:
        st.subheader("Interaction Parameters")
        s1, s2, s3 = st.columns(3)
        with s1:
            contact_type = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
            product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])

        with s2:
            pitch_duration = st.slider("Pitch Duration (Minutes)", min_value=0, max_value=120, value=15)
            pitch_satisfaction = st.select_slider("Pitch Satisfaction Score", options=[1, 2, 3, 4, 5], value=3)

        with s3:
            num_followups = st.selectbox("Number of Follow-ups Made", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=3)

    st.write("---")
    submit_button = st.form_submit_button("Run Purchase Prediction", use_container_width=True, type="primary")

# Prediction
if submit_button:
    # Value conversions
    passport_val = 1 if passport == "Yes" else 0
    own_car_val = 1 if own_car == "Yes" else 0

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

    st.markdown("### Assessment Results")
    
    # Process prediction
    prediction = model.predict(input_data)[0]
    
    # Calculate probability if available in pipeline
    has_proba = hasattr(model, "predict_proba")
    proba = model.predict_proba(input_data)[0][1] if has_proba else None

    r1, r2 = st.columns([1, 2])

    with r1:
        if prediction == 1:
            st.success(":material/local_fire_department: **HIGH PROPENSITY LEAD**")
            if proba is not None:
                st.metric(label="Purchase Probability", value=f"{proba * 100:.1f}%", delta="High Potential")
        else:
            st.error(":material/ac_unit: **LOW PROPENSITY LEAD**")
            if proba is not None:
                st.metric(label="Purchase Probability", value=f"{proba * 100:.1f}%", delta="-Low Potential", delta_color="inverse")

    with r2:
        if prediction == 1:
            st.markdown(
                """
                > **Recommended Action:** Priority lead. Assign a senior account executive and send custom package highlights within 2 hours.
                """
            )
            if proba is not None:
                st.progress(float(proba), text="Model Confidence Gauge")
        else:
            st.markdown(
                """
                > **Recommended Action:** Standard lead nurture workflow. Send automated promotional email sequence rather than immediate direct sales calls.
                """
            )
            if proba is not None:
                st.progress(float(proba), text="Model Confidence Gauge")
