import joblib
import pandas as pd
import time

# Model imports
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("--- Training and Comparing Multiple Models ---")

# --- 1. Load Processed Data ---
try:
    X_train = joblib.load('data/processed/X_train_processed_new.pkl')
    y_train = joblib.load('data/processed/y_train_new.pkl')
    X_test = joblib.load('data/processed/X_test_processed_new.pkl')
    y_test = joblib.load('data/processed/y_test_new.pkl')
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Please run preprocess_new.py first.")
    exit()

# --- 2. Define Models ---
# Using class_weight='balanced' for Logistic Regression and Decision Tree due to imbalance
# For XGBoost, scale_pos_weight is the equivalent parameter. 
# It's calculated as count(negative_class) / count(positive_class)
y_train_df = pd.Series(y_train)
scale_pos_weight = y_train_df.value_counts()[0] / y_train_df.value_counts()[1]

models = {
    "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
}

# --- 3. Train and Evaluate Each Model ---
for name, model in models.items():
    print(f"\n{'='*20}")
    print(f"Training {name}...")
    print(f"{ '='*20}")
    
    start_time = time.time()
    
    # Train the model
    model.fit(X_train, y_train)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"--- Results for {name} ---")
    print(f"Training Time: {training_time:.4f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"{ '='*20}\n")

print("--- Model Comparison Complete ---")
