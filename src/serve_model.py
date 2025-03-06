import logging
import os
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, jwt_required, create_access_token
from datetime import timedelta
import joblib
import pandas as pd
import redis
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
log_file = os.getenv("LOG_FILE", "audit.log")
handler = logging.FileHandler(log_file)
formatter = logging.Formatter("%(asctime)s - %(user)s - %(ip)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Initialize Flask app
app = Flask(__name__)


# Load model
model_path = os.getenv("MODEL_PATH", "models/fraud_detection_model.pkl")
model = joblib.load(model_path)


@app.before_request
def log_request_info():
    try:
        user = get_jwt().get("identity", "anonymous")
    except:
        user = "anonymous"

    logger.info(
        "Request received",
        extra={
            "user": user,
            "ip": request.remote_addr,
            "endpoint": request.endpoint,
            "method": request.method,
            "content_length": request.content_length,
        },
    )


@app.route("/predict", methods=["POST"])
@jwt_required()
def predict():
    try:
        data = request.json
        features = data.get("features")

        # Enhanced logging
        logger.info(f"Received prediction request with features: {features}")

        df = pd.DataFrame([features])
        data_str = json.dumps(df.to_dict(orient="records")[0])

        # Check cache
        cached_result = cache.get(data_str)
        if cached_result:
            logger.info("Cache hit")
            return jsonify(
                {"prediction": int(cached_result.decode()), "source": "cache"}
            )

        # Model prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        # Cache result
        cache.setex(data_str, 3600, str(prediction))

        logger.info(f"Prediction: {prediction}, Probability: {probability}")
        return jsonify(
            {
                "prediction": int(prediction),
                "probability": float(probability),
                "source": "model",
            }
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    app.run(
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", 5000)),
        debug=os.getenv("DEBUG_MODE", "False").lower() == "true",
    )
