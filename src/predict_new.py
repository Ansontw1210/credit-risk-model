import joblib
import pandas as pd
import numpy as np

print("--- Making a prediction with the new model ---")

# --- 1. Load the Preprocessor and Model ---
try:
    preprocessor = joblib.load('models/preprocessor_new.pkl')
    model = joblib.load('models/logistic_regression_model_new.pkl')
    print("Preprocessor and model loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Please run preprocess_new.py and train_new.py first.")
    exit()

# --- 2. Define Sample New Data ---
# This data point is based on the first row of the dataset, where the actual loan_status was 1 (default)
sample_data = {
    'person_age': 22,
    'person_income': 59000,
    'person_home_ownership': 'RENT',
    'person_emp_length': 1.0, # Using 1.0 instead of the outlier 123.0 for a realistic test
    'loan_intent': 'PERSONAL',
    'loan_grade': 'D',
    'loan_amnt': 35000,
    'loan_int_rate': 16.02,
    'loan_percent_income': 0.59,
    'cb_person_default_on_file': 'Y',
    'cb_person_cred_hist_length': 3
}

# What if the person had a better loan grade and lower loan amount?
sample_data_good_risk = {
    'person_age': 22,
    'person_income': 59000,
    'person_home_ownership': 'RENT',
    'person_emp_length': 1.0,
    'loan_intent': 'PERSONAL',
    'loan_grade': 'A',  # Better loan grade
    'loan_amnt': 5000,   # Lower loan amount
    'loan_int_rate': 7.5, # Lower interest rate
    'loan_percent_income': 0.08, # Lower percentage of income
    'cb_person_default_on_file': 'N', # No previous default
    'cb_person_cred_hist_length': 3
}


# --- 3. Make Prediction ---
def predict_risk(data):
    # Convert dict to DataFrame
    df = pd.DataFrame([data])
    
    # Ensure correct column order and types (optional but good practice)
    # This step is handled by the scikit-learn pipeline itself
    
    # Preprocess the data
    data_processed = preprocessor.transform(df)
    
    # Predict
    prediction = model.predict(data_processed)
    prediction_proba = model.predict_proba(data_processed)
    
    # Interpret result
    status = 'Default' if prediction[0] == 1 else 'Non-Default'
    
    print(f"\nPrediction for sample data: {status}")
    print(f"Probabilities (0: Non-Default, 1: Default): {prediction_proba[0]}")

# Predict for both scenarios
print("\n--- Scenario 1: High Risk Applicant (based on original data) ---")
predict_risk(sample_data)

print("\n--- Scenario 2: Low Risk Applicant (modified data) ---")
predict_risk(sample_data_good_risk)
