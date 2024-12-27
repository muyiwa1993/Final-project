from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the scaler, model, and label encoders
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_cluster():
    try:
        # Get JSON data from the request
        data = request.json
        input_df = pd.DataFrame(data)

        # Apply label encoding to the categorical columns
        categorical_columns = [
            'Gender', 'Marital Status', 'Education Level', 'Geographic Information', 
            'Occupation', 'Behavioral Data', 'Policy Type', 'Customer Preferences', 
            'Preferred Communication Channel', 'Preferred Contact Time', 
            'Preferred Language', 'Segmentation Group'
        ]

        for col in categorical_columns:
            if col in input_df.columns:
                input_df[col] = label_encoders[col].transform(input_df[col])

        # Scale the input data
        X_scaled = scaler.transform(input_df)

        # Predict clusters
        clusters = kmeans.predict(X_scaled)
        return jsonify({'clusters': clusters.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
