# streamlit_app.py
#from xgboost import XGBClassifier
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🔄",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open("xgb_churn_model.pkl", "rb") as f:
            model = pickle.load(f)
            st.session_state['feature_names'] = model.feature_names_in_
            return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

if model is None:
    st.error("Failed to load the model. Please check if the model file exists and is valid.")
    st.stop()

# Title and description
st.title("Customer Churn Prediction")
st.markdown("""
This application predicts whether a customer is likely to churn based on various features.
Enter the customer information below to get a prediction.
""")

# Create input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        
    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        
    col3, col4 = st.columns(2)
    
    with col3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])
        
    with col4:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=1000.0, value=50.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0)
        
    submitted = st.form_submit_button("Predict Churn")

# Make prediction when form is submitted
if submitted:
    try:
        # Prepare input data
        data = {
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': str(total_charges)  # Convert to string to match training data
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Convert categorical variables
        binary_map = {
            'Yes': 1,
            'No': 0,
            'No internet service': 0,
            'No phone service': 0
        }
        
        binary_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
                         'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                         'StreamingTV', 'StreamingMovies', 'MultipleLines']
        
        for col in binary_columns:
            input_df[col] = input_df[col].replace(binary_map)
            
        # Map gender
        input_df['gender'] = input_df['gender'].map({'Female': 0, 'Male': 1})
        
        # Map Internet Service
        input_df['InternetService'] = input_df['InternetService'].map({'No': 0, 'DSL': 1, 'Fiber optic': 2})
        
        # Map Contract
        input_df['Contract'] = input_df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
        
        # Map PaymentMethod
        input_df['PaymentMethod'] = input_df['PaymentMethod'].map({
            'Electronic check': 0,
            'Mailed check': 1,
            'Bank transfer (automatic)': 2,
            'Credit card (automatic)': 3
        })
        
        # Convert TotalCharges to numeric
        input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'])
        
        # Ensure all columns are in the correct order
        if 'feature_names' in st.session_state:
            input_df = input_df[st.session_state['feature_names']]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        # Convert probability to Python float
        prob_float = float(probability)
        
        # Display result
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error("⚠️ Customer is likely to churn!")
            else:
                st.success("✅ Customer is likely to stay!")
                
        with col2:
            st.subheader("Churn Probability")
            st.progress(prob_float)
            st.write(f"Probability of churning: {prob_float:.2%}")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.write("Debug information:")
        st.write("Input data shape:", input_df.shape)
        st.write("Input data columns:", input_df.columns.tolist())
        if 'feature_names' in st.session_state:
            st.write("Model features:", st.session_state['feature_names'].tolist())

