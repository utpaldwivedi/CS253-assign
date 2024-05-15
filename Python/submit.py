import numpy as np 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

# Read the data
train_data = pd.read_csv('/kaggle/input/who-is-the-real-winner/train.csv')
test_data = pd.read_csv('/kaggle/input/who-is-the-real-winner/test.csv')
sample_submission = pd.read_csv('/kaggle/input/who-is-the-real-winner/sample_submission.csv')

# Define columns of interest
columns = ['Party', 'Criminal Case', 'Total Assets', 'Liabilities', 'state']

# Extract features and target variable
X_train = train_data[columns]
X_test = test_data[columns]
y_train = train_data['Education']

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode categorical features
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()
for col in X_train.columns:
    X_train_encoded[col] = label_encoder.fit_transform(X_train[col])
    X_test_encoded[col] = label_encoder.fit_transform(X_test[col])

# Encode target variable
y_train_encoded = label_encoder.fit_transform(y_train)

# Define the hyperparameters to tune
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=1)

# Initialize GridSearchCV
grid_search = GridSearchCV(dt_classifier, param_grid, cv=5, scoring='f1_micro')

# Perform Grid Search to find the best parameters
grid_search.fit(X_train_encoded, y_train_encoded)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Initialize the model with the best parameters
best_model = DecisionTreeClassifier(**best_params, random_state=1)

# Train the model with the full training data
best_model.fit(X_train_encoded, y_train_encoded)

# Make predictions on the test data
y_pred_encoded = best_model.predict(X_test_encoded)

# Decode the predictions
y_pred_original = label_encoder.inverse_transform(y_pred_encoded)

# Create a submission DataFrame
submission = pd.DataFrame({'ID': test_data['ID'], 'Education': y_pred_original})

# Save the submission to a CSV file
file_path = 'submission.csv'
submission.to_csv(file_path, index=False)
