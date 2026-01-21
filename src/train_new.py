import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

print("Starting model training for the new dataset...")

# --- 1. Load Processed Data ---
X_train_processed = joblib.load('data/processed/X_train_processed_new.pkl')
y_train = joblib.load('data/processed/y_train_new.pkl')
X_test_processed = joblib.load('data/processed/X_test_processed_new.pkl')
y_test = joblib.load('data/processed/y_test_new.pkl')

print("Data loaded successfully.")

# --- 2. Initialize and Train the Model ---
# Using class_weight='balanced' because the EDA showed class imbalance.
model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
print("Training the Logistic Regression model...")
model.fit(X_train_processed, y_train)
print("Model training finished.")

# --- 3. Evaluate the Model ---
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test_processed)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# --- 4. Save the Trained Model ---
joblib.dump(model, 'models/logistic_regression_model_new.pkl')

print("\nNew model training complete. Trained model 'logistic_regression_model_new.pkl' has been saved.")
