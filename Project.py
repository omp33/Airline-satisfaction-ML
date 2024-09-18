import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras

# Load the dataset
data = pd.read_csv('satisfaction train.csv')

# Drop unwanted columns
data = data.drop(columns=['id'], axis=1)

# One hot encoding for categorical features
d1 = pd.get_dummies(data[['Gender']],prefix='Gender')
d2 = pd.get_dummies(data[['Customer Type']],prefix='Customer Type')
d3 = pd.get_dummies(data[['Type of Travel']],prefix='Type of Travel')
d4 = pd.get_dummies(data[['Class']],prefix='Class')

# Concatenate the encoded data with the original dataframe
data = pd.concat([data, d1, d2, d3, d4], axis=1)

# Drop the original categorical columns
data = data.drop(columns=['Gender', 'Customer Type', 'Type of Travel', 'Class'], axis=1)

# Separate the target column, here 'satisfaction' is binary so we can convert it into a single target column.
data['satisfaction'] = data['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})

# Split the dataset into features and target
X = data.drop(columns=['satisfaction'])  # Features
y = data['satisfaction']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
