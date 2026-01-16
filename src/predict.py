import joblib
import pandas as pd

# Load the preprocessor and the model
preprocessor = joblib.load('models/preprocessor.pkl')
model = joblib.load('models/logistic_regression_model.pkl')

# Example new data point (as a dictionary)
new_data = {
    'existing_checking_account': 'A11',
    'duration_in_month': 6,
    'credit_history': 'A34',
    'purpose': 'A43',
    'credit_amount': 1169,
    'savings_account_bonds': 'A65',
    'present_employment_since': 'A75',
    'installment_rate_percentage': 4,
    'personal_status_sex': 'A93',
    'other_debtors_guarantors': 'A101',
    'present_residence_since': 4,
    'property': 'A121',
    'age_in_years': 67,
    'other_installment_plans': 'A143',
    'housing': 'A152',
    'number_credits_at_bank': 2,
    'job': 'A173',
    'number_people_liable': 1,
    'telephone': 'A192',
    'foreign_worker': 'A201'
}

# Convert the new data to a DataFrame
new_df = pd.DataFrame([new_data])

# Preprocess the new data
new_data_processed = preprocessor.transform(new_df)

# Make a prediction
prediction = model.predict(new_data_processed)
prediction_proba = model.predict_proba(new_data_processed)

# Print the prediction
# The prediction is 0 for "Good" and 1 for "Bad" credit risk.
print(f"Prediction for the new data: {'Good' if prediction[0] == 0 else 'Bad'}")
print(f"Prediction probability: {prediction_proba}")
