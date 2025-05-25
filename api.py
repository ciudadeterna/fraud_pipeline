from flask import Flask, request, jsonify
import joblib
import pandas as pd
import mlflow

app = Flask(__name__)
model = joblib.load('fraud_model.pkl')

API_KEY = "RONALDINHO"  
@app.route('/predict', methods=['POST'])
def predict():
    token = request.headers.get("Authorization")
    if token != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(force=True)
    df = pd.DataFrame([data])

    prob = model.predict_proba(df)[0][1]
    prediction = model.predict(df)[0]

    with mlflow.start_run(run_name="prediction", nested=True):
        mlflow.log_metric("fraud_probability", prob)
        mlflow.log_param("Amount", data.get("Amount", -1))
        mlflow.set_tag("prediction", prediction)

    return jsonify({
        'fraud_probability': prob,
        'is_fraud': int(prediction)
    })

if __name__ == '__main__':
    app.run(debug=True)
