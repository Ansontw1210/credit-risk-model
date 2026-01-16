import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Load the processed data
X_train_processed = joblib.load('data/processed/X_train_processed.pkl')
X_test_processed = joblib.load('data/processed/X_test_processed.pkl')
y_train = joblib.load('data/processed/y_train.pkl')
y_test = joblib.load('data/processed/y_test.pkl')

# Initialize and train the Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_processed, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_processed)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Save the trained model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/logistic_regression_model.pkl')

print("Model training complete. Trained model saved.")
