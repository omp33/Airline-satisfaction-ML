import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load your dataset (assuming it's preprocessed as shown earlier)
data = pd.read_csv('satisfaction train.csv')

# Handling NaN values (example with 'Arrival Delay in Minutes')
data['Arrival Delay in Minutes'] = data['Arrival Delay in Minutes'].fillna(data['Arrival Delay in Minutes'].mean())

# One hot encoding and target encoding
d1 = pd.get_dummies(data[['Gender']], prefix='Gender')
d2 = pd.get_dummies(data[['Customer Type']], prefix='Customer Type')
d3 = pd.get_dummies(data[['Type of Travel']], prefix='Type of Travel')
d4 = pd.get_dummies(data[['Class']], prefix='Class')
data = pd.concat([data, d1, d2, d3, d4], axis=1)
data = data.drop(columns=['Gender', 'Customer Type', 'Type of Travel', 'Class'], axis=1)
data['satisfaction'] = data['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})

# Features and target
X = data.drop(columns=['satisfaction'])  # Features
y = data['satisfaction']  # Target

# Fix column names by replacing spaces or special characters with underscores
X.columns = X.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize the LGBMClassifier with the given settings
lgbm = lgb.LGBMClassifier(boosting_type='gbdt',
                          class_weight=None,
                          colsample_bytree=1.0,
                          importance_type='split',
                          learning_rate=0.1,
                          max_depth=-1,
                          min_child_samples=20,
                          min_child_weight=0.001,
                          min_split_gain=0.0,
                          n_estimators=100,
                          n_jobs=-1,
                          num_leaves=31,
                          objective=None,
                          random_state=42,
                          reg_alpha=0.0,
                          reg_lambda=0.0,
                          subsample=1.0,
                          subsample_for_bin=200000,
                          subsample_freq=0)

# Lists to store metrics for each fold
accuracy_scores = []
confusion_matrices = []

# Stratified K-Fold Cross-Validation
for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train the model
    lgbm.fit(X_train, y_train)
    
    # Predict on the test set for this fold
    y_pred = lgbm.predict(X_test)
    
    # Calculate and print accuracy for this fold
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Fold {fold} Accuracy: {accuracy}")
    accuracy_scores.append(accuracy)
    
    # Print classification report for this fold
    print(f"Fold {fold} Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Fold {fold} Confusion Matrix:\n{cm}\n")
    confusion_matrices.append(cm)

# Overall metrics after 10 folds
print(f"Average Accuracy over 10 folds: {np.mean(accuracy_scores)}")
