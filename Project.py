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
data = data.drop(columns=['#','id'])

#Mapping targets to 1 for satistfied and 0 for non satistfied

data['satisfaction'] = data['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})

# Split the dataset into features and target

X = data.drop(columns=['satisfaction'])
y = data['satisfaction']




