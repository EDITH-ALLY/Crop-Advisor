from flask import Flask, request
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = pickle.load(open('model/model.pkl', 'rb'))

@app.route('/')
def home():
    return "Crop Advisor Backend Running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = [float(x) for x in request.form.values()]
        final = np.array([values])

        prediction = model.predict(final)[0]

        return f"Recommended Crop: {prediction}"
    except:
        return "Error in prediction"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
