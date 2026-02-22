from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask application
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('parkinsons_model.joblib')
scaler = joblib.load('scaler.joblib')

print("Model 'parkinsons_model.joblib' loaded successfully.")
print("Scaler 'scaler.joblib' loaded successfully.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    if 'features' not in data:
        return jsonify({'error': 'Missing "features" key in request body.'}), 400

    features = data['features']
    if not isinstance(features, list) or len(features) != 22:
        return jsonify({'error': '"features" should be a list of 22 numerical values.'}), 400

    # Convert features to numpy array and reshape for scaling
    features_array = np.array(features).reshape(1, -1)

    # Scale the features using the loaded scaler
    features_scaled = scaler.transform(features_array)

    # Make prediction using the loaded model
    prediction = model.predict(features_scaled)

    # Return the prediction as JSON
    return jsonify({'prediction': int(prediction[0])}), 200


if __name__ == "__main__":
    app.run(debug=True)
