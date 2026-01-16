import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import os

# Create directories if they don't exist
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Define column names for the dataset
column_names = [
    'existing_checking_account', 'duration_in_month', 'credit_history', 'purpose',
    'credit_amount', 'savings_account_bonds', 'present_employment_since',
    'installment_rate_percentage', 'personal_status_sex', 'other_debtors_guarantors',
    'present_residence_since', 'property', 'age_in_years', 'other_installment_plans',
    'housing', 'number_credits_at_bank', 'job', 'number_people_liable',
    'telephone', 'foreign_worker', 'credit_risk'
]

# Load the dataset
df = pd.read_csv('data/german.data', sep=' ', header=None, names=column_names)

# Map target variable
df['credit_risk'] = df['credit_risk'].map({1: 0, 2: 1})

# Separate features and target
X = df.drop('credit_risk', axis=1)
y = df['credit_risk']

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Create the preprocessing pipelines for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and transform the training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform the test data
X_test_processed = preprocessor.transform(X_test)

# Save the processed data
joblib.dump(X_train_processed, 'data/processed/X_train_processed.pkl')
joblib.dump(X_test_processed, 'data/processed/X_test_processed.pkl')
joblib.dump(y_train, 'data/processed/y_train.pkl')
joblib.dump(y_test, 'data/processed/y_test.pkl')

# Save the preprocessor
joblib.dump(preprocessor, 'models/preprocessor.pkl')

print("Preprocessing complete. Processed data and preprocessor saved.")
