import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import os
import numpy as np

print("Starting preprocessing for the new dataset...")

# --- 1. Load Data ---
df = pd.read_csv('data/credit_risk_dataset.csv')

# --- 2. Clean Data ---
# Remove unrealistic values. Age > 100 and employment length > 60 are likely errors.
df_cleaned = df[df['person_age'] <= 100]
df_cleaned = df_cleaned[df_cleaned['person_emp_length'] <= 60]

# --- 3. Separate Features and Target ---
X = df_cleaned.drop('loan_status', axis=1)
y = df_cleaned['loan_status']

# --- 4. Define Feature Types ---
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"Identified numerical features: {numerical_features}")
print(f"Identified categorical features: {categorical_features}")

# --- 5. Create Preprocessing Pipelines ---
# Pipeline for numerical features: impute missing values with median, then scale.
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline for categorical features: impute missing values with 'missing', then one-hot encode.
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# --- 6. Create the Master Preprocessor ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- 7. Split Data ---
# Using stratify on y because the EDA showed class imbalance.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 8. Fit and Transform Data ---
# Fit the preprocessor on the training data and transform it
X_train_processed = preprocessor.fit_transform(X_train)

# Transform the test data
X_test_processed = preprocessor.transform(X_test)

# --- 9. Save Processed Data and Preprocessor ---
# Create directories if they don't exist
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Save the processed data splits
joblib.dump(X_train_processed, 'data/processed/X_train_processed_new.pkl')
joblib.dump(X_test_processed, 'data/processed/X_test_processed_new.pkl')
joblib.dump(y_train, 'data/processed/y_train_new.pkl')
joblib.dump(y_test, 'data/processed/y_test_new.pkl')

# Save the fitted preprocessor
joblib.dump(preprocessor, 'models/preprocessor_new.pkl')

print("\nPreprocessing complete.")
print("Processed data and the new preprocessor have been saved.")
print(f"Training data shape: {X_train_processed.shape}")
print(f"Test data shape: {X_test_processed.shape}")
