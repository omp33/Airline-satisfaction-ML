from flask import Flask, request, jsonify
import pandas as pd
import pickle

# Load the trained LightGBM model
model_filename = 'lgbm_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

# Define a route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.get_json()

    # Create a DataFrame from the input data
    input_data = pd.DataFrame(data, index=[0])

    # One-hot encode categorical features (same as training)
    d1 = pd.get_dummies(input_data[['Gender']], prefix='Gender')
    d2 = pd.get_dummies(input_data[['Customer Type']], prefix='Customer Type')
    d3 = pd.get_dummies(input_data[['Type of Travel']], prefix='Type of Travel')
    d4 = pd.get_dummies(input_data[['Class']], prefix='Class')

    # Concatenate the one-hot encoded variables with the original DataFrame
    input_data = pd.concat([input_data, d1, d2, d3, d4], axis=1)
    
    # Drop original categorical columns
    input_data = input_data.drop(columns=['Gender', 'Customer Type', 'Type of Travel', 'Class'], axis=1)

    # Fix column names by replacing spaces or special characters with underscores
    input_data.columns = input_data.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

    # Ensure all columns used during training are present in the input data
    for col in model.feature_name_:
        if col not in input_data.columns:
            input_data[col] = 0  # Assign 0 if the column is missing

    # Make prediction
    prediction = model.predict(input_data)

    # Convert prediction to human-readable format
    result = 'satisfied' if prediction[0] == 1 else 'neutral or dissatisfied'

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)