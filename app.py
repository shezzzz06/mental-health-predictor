from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# load your model
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

model = joblib.load(model_path)

@app.route("/")
def home():
    return "API is working!"

def match_count(user_symptoms, required_symptoms):
    return sum(1 for sym in required_symptoms if sym in user_symptoms)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    symptoms = data.get("symptoms", [])

    scores = {
        "Bipolar disorder": match_count(symptoms, [
            "Inflated self-esteem",
            "Distractibility",
            "Increase in goal-directed activity",
            "Excessive involvement in activities with high potential for painful consequences",
            "Racing thoughts",
            "Decreased need for sleep",
            "More talkative than usual"

        ]),
        "Depression": match_count(symptoms, [
            "Depressed mood",
            "Persistent sadness or low mood",
            "Loss of interest or pleasure in activities",
            "Fatigue or loss of energy",
            "Thoughts of suicide"
        ]),
        "Anxiety disorder": match_count(symptoms, [
            "Excessive worry or fear",
             "Restlessness","Irritability",
            "Sleep disturbance"
        ]),
        "Schizophrenia": match_count(symptoms, [
            "Disorganized thinking or speech","Delusions",
             "Hallucinations","Catatonic behavior",
            "Diminished emotional expression"
        ]),
        "PTSD": match_count(symptoms, [
            "Experiencing traumatic event", "Hypervigilance",
            "Exaggerated startle response",
            "Avoidance of reminders of traumatic event"
        ])
    }

    # pick highest score
    best_match = max(scores, key=scores.get)

    if scores[best_match] == 0:
        result = "No clear disorder detected"
    else:
        result = best_match

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)