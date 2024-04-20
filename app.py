from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle
import joblib
from tensorflow.keras.models import load_model

# Load the saved model using load_model()
loaded_model = load_model('my_model')


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        result = request.form.to_dict()
        print("Result of Form -------> ",result)
        input=list(result.values())
        
        print("Input List --------> ",input)
        input_array = np.array(input).reshape(1, -1)  # Reshape to a 2D array with a single feature

        loaded_scaler = joblib.load('scaler.pkl')

        scaled_input = loaded_scaler.transform(input_array)
        print("Scaled Input ---------->",scaled_input)

        
        predictions = loaded_model.predict(scaled_input)
        print("Prediction ----------> ",predictions)
       
        if predictions[0][0]>0.5:
            return render_template('predict.html',prediction_text="Student will Persist")
        else:
            return render_template('predict.html',prediction_text="Student will Not Persist")

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
