import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import shap

# Load the dataset
data = pd.read_csv('satisfaction train.csv')

# Check for NaN values
print("Checking for NaN values in the dataset:")
print(data.isnull().sum())

# Drop unwanted columns
data = data.drop(columns=['id'], axis=1)

# One hot encoding for categorical features
d1 = pd.get_dummies(data[['Gender']], prefix='Gender')
d2 = pd.get_dummies(data[['Customer Type']], prefix='Customer Type')
d3 = pd.get_dummies(data[['Type of Travel']], prefix='Type of Travel')
d4 = pd.get_dummies(data[['Class']], prefix='Class')

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

# Creating the model with reduced regularization, dropout, and added gradient clipping
model = keras.models.Sequential([

    keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],),
                       kernel_regularizer=regularizers.l2(0.001)),  # Reduced L2 regularization
    BatchNormalization(),
    Dropout(0.2),  # Reduced dropout

    keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),

    keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),

    keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),

    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with a lower learning rate and gradient clipping
opt = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Add early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(X_train_scaled, y_train, 
                    validation_split=0.2, 
                    epochs=100, 
                    callbacks=[early_stopping],
                    batch_size=32)

# Evaluate the model
y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Plot training & validation accuracy and loss
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# SHAP interpretation for mean SHAP plot
explainer = shap.KernelExplainer(model.predict, X_train_scaled[:100])  # Using first 100 rows for speed
shap_values = explainer.shap_values(X_test_scaled[:100])  # Speed up using 100 rows

# SHAP summary plot (mean SHAP value plot)
shap.summary_plot(shap_values, X_test_scaled[:100], plot_type="bar")
