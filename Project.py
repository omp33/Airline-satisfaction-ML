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

#Drop unwanted columns
data = data.drop(columns=['id'],axis=1)

# One hot encoding for featurs and targets.

data = pd.get_dummies(data, columns=['Gender','Customer Type','Type of Travel','Class','satisfaction',])


# Split the dataset into features and target
X = data.drop(columns=['satisfaction'])
y = data['satisfaction']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#





