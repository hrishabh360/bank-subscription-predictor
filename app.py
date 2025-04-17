from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load model and preprocessors
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Extract numeric and categorical features
    numeric = np.array([[data['age'], data['balance']]])
    categoric = np.array([[data['job'], data['marital'], data['education'], data['default'], data['housing'], data['loan']]])

    # Preprocess
    numeric_scaled = scaler.transform(numeric)
    categoric_encoded = encoder.transform(categoric).toarray()
    final_input = np.concatenate((numeric_scaled, categoric_encoded), axis=1)

    # Predict
    prediction = model.predict(final_input)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
