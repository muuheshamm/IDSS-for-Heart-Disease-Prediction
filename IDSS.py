# IDSS.py

from keras.models import load_model
import pandas as pd

def load_trained_model(model_path):
    """
    Load the pre-trained neural network model.
    
    Parameters:
    model_path (str): Path to the saved model file.
    
    Returns:
    model: Loaded neural network model.
    """
    return load_model(model_path)

def prepare_input_data(patient_data):
    """
    Prepare the input data for prediction.
    
    Parameters:
    patient_data (dict): Dictionary containing patient data.
    
    Returns:
    DataFrame: Prepared input data as a pandas DataFrame.
    """
    return pd.DataFrame([patient_data])

def make_prediction(model, input_data):
    """
    Make predictions using the loaded model.
    
    Parameters:
    model: Loaded neural network model.
    input_data (DataFrame): Prepared input data.
    
    Returns:
    tuple: Predicted probabilities and predicted class.
    """
    predicted_prob = model.predict(input_data)
    predicted_class = predicted_prob.argmax(axis=-1)
    return predicted_prob, predicted_class

def interpret_prediction(predicted_class):
    """
    Interpret the prediction result and provide decision support.
    
    Parameters:
    predicted_class (int): Predicted class label.
    
    Returns:
    str: Interpretation and advice based on the predicted class.
    """
    if predicted_class == 1:
        return "High risk of heart disease. It is recommended to consult a cardiologist immediately."
    else:
        return "Low risk of heart disease. Maintain a healthy lifestyle and regular check-ups."

def main():
    # Define the path to the saved model
    model_path = '/Users/mohamedhesham/Downloads/dss project/nn_model.weights.h5'
    
    # Load the trained model
    model = load_trained_model(model_path)
    
    # New patient data
    new_patient_data = {
        'age': 52,
        'sex': 1,
        'cp': 0,
        'trestbps': 125,
        'chol': 212,
        'fbs': 0,
        'restecg': 1,
        'thalach': 168,
        'exang': 0,
        'oldpeak': 1.0,
        'slope': 2,
        'ca': 2,
        'thal': 3
    }
    
    # Prepare the input data
    input_data = prepare_input_data(new_patient_data)
    
    # Make predictions
    predicted_prob, predicted_class = make_prediction(model, input_data)
    
    # Print the results
    print(f"Predicted probabilities: {predicted_prob}")
    print(f"Predicted class: {predicted_class}")
    
    # Provide interpretation and decision support
    interpretation = interpret_prediction(predicted_class[0])
    print(interpretation)

if __name__ == "__main__":
    main()