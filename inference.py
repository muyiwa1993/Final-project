import os
import pickle
import joblib
import numpy as np
import json

# Load artifacts
def model_fn(model_dir):
    """
    Load the model artifacts from the given model directory.
    """
    try:
        # Paths to the files in the model directory
        model_path = os.path.join(model_dir, "kmeans_model.pkl")
        label_encoders_path = os.path.join(model_dir, "label_encoders.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")

        # Load the model, encoders, and scaler
        with open(model_path, "rb") as f:
            model = joblib.load(f)
        with open(label_encoders_path, "rb") as f:
            label_encoders = joblib.load(f)
        with open(scaler_path, "rb") as f:
            scaler = joblib.load(f)

        return {"model": model, "label_encoders": label_encoders, "scaler": scaler}
    except Exception as e:
        raise ValueError(f"Error loading model artifacts: {e}")

def input_fn(request_body, request_content_type):
    """
    Parse and validate the input data.
    """
    try:
        if request_content_type == "application/json":
            return json.loads(request_body)  # Expecting JSON input
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")
    except Exception as e:
        raise ValueError(f"Error parsing input: {e}")

def predict_fn(input_data, model_artifacts):
    """
    Generate predictions based on the input data and model artifacts.
    """
    try:
        model = model_artifacts["model"]
        label_encoders = model_artifacts["label_encoders"]
        scaler = model_artifacts["scaler"]

        # Preprocess the input data
        categorical_columns = ['Gender', 'Marital Status', 'Education Level']  # Add all relevant columns
        for col in categorical_columns:
            if col in input_data:
                input_data[col] = label_encoders[col].transform([input_data[col]])[0]
        
        # Extract numerical features and scale them
        numerical_features = [value for key, value in input_data.items() if key not in categorical_columns]
        numerical_features = np.array(numerical_features).reshape(1, -1)
        scaled_features = scaler.transform(numerical_features)

        # Predict using the model
        cluster = model.predict(scaled_features)
        return {"cluster": int(cluster[0])}
    except Exception as e:
        raise ValueError(f"Error during prediction: {e}")

def output_fn(prediction, response_content_type):
    """
    Format the output for the response.
    """
    try:
        if response_content_type == "application/json":
            return json.dumps(prediction)
        else:
            raise ValueError(f"Unsupported response content type: {response_content_type}")
    except Exception as e:
        raise ValueError(f"Error formatting output: {e}")
