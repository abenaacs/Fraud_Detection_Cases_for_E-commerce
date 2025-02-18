import logging
from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)

# Load the trained model
model = joblib.load("models/fraud_detection_model.pkl")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse JSON input
        data = request.json
        features = data["features"]
        logging.info(f"Received request with features: {features}")

        # Convert to DataFrame
        df = pd.DataFrame([features])

        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]  # Probability of fraud

        # Log prediction
        logging.info(f"Prediction: {prediction}, Probability: {probability}")

        return jsonify(
            {"prediction": int(prediction), "probability": float(probability)}
        )
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 400


@app.route("/fraud-insights", methods=["GET"])
def fraud_insights():
    try:
        # Load processed data
        fraud_data = pd.read_csv("data/processed_fraud_data.csv")

        # Summary statistics
        total_transactions = len(fraud_data)
        fraud_cases = fraud_data["class"].sum()
        fraud_percentage = (fraud_cases / total_transactions) * 100

        # Trends over time
        fraud_data["purchase_time"] = pd.to_datetime(fraud_data["purchase_time"])
        fraud_trends = fraud_data.groupby(fraud_data["purchase_time"].dt.date)[
            "class"
        ].sum()

        return jsonify(
            {
                "total_transactions": total_transactions,
                "fraud_cases": int(fraud_cases),
                "fraud_percentage": round(fraud_percentage, 2),
                "fraud_trends": fraud_trends.to_dict(),
            }
        )
    except Exception as e:
        logging.error(f"Error fetching fraud insights: {str(e)}")
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
