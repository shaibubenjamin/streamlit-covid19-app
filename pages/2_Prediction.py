import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from data_loader import get_preprocessed_data

# Page configuration
st.set_page_config(layout="wide")
st.title("🔮COVID-19 Test Prediction")
st.markdown("""
### Complete the form below to predict COVID-19 test result
""")

@st.cache_resource
def load_model():
    """Load and cache the trained model"""
    try:
        X_encoded, y, _ = get_preprocessed_data()
        
        # Handle class imbalance
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X_encoded, y)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42
        )
        model.fit(X_resampled, y_resampled)
        return model, X_encoded.columns
    
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

# Load model
model, feature_columns = load_model()

if model is None:
    st.stop()

# Input form - Expanded version
with st.form("prediction_form"):
    st.subheader("Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Personal Information
        st.markdown("#### Personal Information")
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=30)
        sex = st.selectbox("Gender", ["MALE", "FEMALE", "UNKNOWN"])
        
        # Symptoms
        st.markdown("#### Symptoms")
        fever = st.radio("Fever", ["YES", "NO", "UNKNOWN"], horizontal=True)
        cough = st.radio("Cough", ["YES", "NO", "UNKNOWN"], horizontal=True)
        sore_throat = st.radio("Sore Throat", ["YES", "NO", "UNKNOWN"], horizontal=True)
        fatigue = st.radio("Fatigue", ["YES", "NO", "UNKNOWN"], horizontal=True)
    
    with col2:
        # Additional Symptoms
        st.markdown("#### Additional Symptoms")
        shortness_breath = st.radio("Shortness of Breath", ["YES", "NO", "UNKNOWN"], horizontal=True)
        headache = st.radio("Headache", ["YES", "NO", "UNKNOWN"], horizontal=True)
        muscle_pain = st.radio("Muscle Pain", ["YES", "NO", "UNKNOWN"], horizontal=True)
        loss_taste = st.radio("Loss of Taste/Smell", ["YES", "NO", "UNKNOWN"], horizontal=True)
        
        # Comorbidities
        st.markdown("#### Comorbidities")
        diabetes = st.radio("Diabetes", ["YES", "NO", "UNKNOWN"], horizontal=True)
        hypertension = st.radio("Hypertension", ["YES", "NO", "UNKNOWN"], horizontal=True)
        
    submitted = st.form_submit_button("PREDICT TEST RESULT", type="primary")

if submitted:
    # Prepare input data
    input_data = {
        'Age': age,
        'Sex': sex,
        'Fever': fever,
        'Cough': cough,
        'Sore Throat': sore_throat,
        'Fatigue': fatigue,
        'Shortness of Breath': shortness_breath,
        'Headache': headache,
        'Muscle Pain': muscle_pain,
        'Loss of Taste/Smell': loss_taste,
        'Diabetes': diabetes,
        'Hypertension': hypertension
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # One-hot encode
    cat_cols = [col for col in input_df.columns if input_df[col].dtype == 'object']
    input_encoded = pd.get_dummies(input_df, columns=cat_cols)
    
    # Ensure all features exist
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Make prediction
    prediction = model.predict(input_encoded[feature_columns])
    
    # Display results
    st.markdown("---")
    st.subheader("Prediction Result")
    
    if prediction[0] == 1:
        st.error("""
        POSITIVE RESULT 🦠
        The model predicts a positive COVID-19 test result.
        """)
    else:
        st.success("""
        NEGATIVE RESULT ✅
        The model predicts a negative COVID-19 test result.
        """)
