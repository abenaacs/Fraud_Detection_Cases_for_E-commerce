# src/kafka_consumer.py
from kafka import KafkaConsumer
import json
from model_builder import load_model  # Your existing model loader


def process_transaction():
    consumer = KafkaConsumer(
        "fraud-transactions",
        bootstrap_servers="localhost:9092",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    model = load_model()  # Load your trained model

    for message in consumer:
        transaction = message.value
        prediction = model.predict([list(transaction.values())])
        print(f"Transaction {transaction} is {'Fraud' if prediction else 'Legit'}")
