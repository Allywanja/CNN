import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Define the root route for home page
@flask_app.route("/")
def home():
    return render_template("index.html")

# Define the route for prediction when form is submitted
@flask_app.route("/predict", methods=["POST"])
def predict():
    # Getting form data and converting it to float
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    
    # Make prediction using the model
    prediction = model.predict(features)
    
    # Return the result to the user
    return render_template("index.html", prediction_text="The flower species is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)
