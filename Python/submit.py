import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Reading Train and Test Files
train_file_path = '/kaggle/input/who-is-the-real-winner/train.csv'
test_file_path = '/kaggle/input/who-is-the-real-winner/test.csv'
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Mapping for Party and State based on train data
party_mapping = {party: idx for idx, party in enumerate(train_data['Party'].unique())}
state_mapping = {state: idx for idx, state in enumerate(train_data['state'].unique())}

# Apply mappings to both train and test data
train_data['Party'] = train_data['Party'].map(party_mapping)
train_data['state'] = train_data['state'].map(state_mapping)
test_data['Party'] = test_data['Party'].map(party_mapping)
test_data['state'] = test_data['state'].map(state_mapping)

def convert_asset_to_int(asset_str):
    asset_parts = asset_str.strip().split()
    numeric_value = int(asset_parts[0]) 
    
    unit = asset_parts[-1]
    if unit == 'Crore+':
        multiplier = 10000000  
    elif unit == 'Lac+':
        multiplier = 100000  
    elif unit == 'Thou+':
        multiplier = 1000  
    else:
        multiplier = 1  

    integer_value = numeric_value * multiplier
    return integer_value

# Apply asset conversion to both train and test data
train_data['Total Assets'] = train_data['Total Assets'].apply(convert_asset_to_int)
train_data['Liabilities'] = train_data['Liabilities'].apply(convert_asset_to_int)
test_data['Total Assets'] = test_data['Total Assets'].apply(convert_asset_to_int)
test_data['Liabilities'] = test_data['Liabilities'].apply(convert_asset_to_int)

# Encoding Education column to numerical labels based on train data
label_encoder = LabelEncoder()
train_data['Education'] = label_encoder.fit_transform(train_data['Education'])

# Define features and target
features = ['Party', 'Criminal Case', 'Total Assets', 'Liabilities', 'state']
X = train_data[features]
y = train_data['Education']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5,10,15, 20,25 ,30,50,100],
    'min_samples_split': [2,3,4, 5, 10,15,20,50],
    'min_samples_leaf': [1, 2,3, 4,5]
}

# Initialize the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=1)

# Initialize GridSearchCV
grid_search = GridSearchCV(dt_classifier, param_grid, cv=5, scoring='f1_micro')

# Perform Grid Search to find the best parameters
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Initialize the model with the best parameters
best_model = DecisionTreeClassifier(**best_params, random_state=1)

# Train the model with the full training data
best_model.fit(X_train, y_train)

# Make predictions on the validation data using the best model
val_predictions = best_model.predict(X_val)

# Calculate F1 score on the validation data
f1 = f1_score(y_val, val_predictions, average='weighted')
print("Best F1 Score on Validation Set:", f1)

# Make predictions on the test data using the best model
X_test = test_data[features]
test_predictions = best_model.predict(X_test)

# Decode the predictions
predicted_education = label_encoder.inverse_transform(test_predictions)

# Prepare submission file with ID and predicted Education
submission_df = pd.DataFrame({
    'ID': test_data['ID'],
    'Education': predicted_education  # Use the string-format predictions
})

# Save submission file
submission_file_path = 'submission.csv'
submission_df.to_csv(submission_file_path, index=False)
