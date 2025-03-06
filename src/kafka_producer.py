# src/kafka_producer.py
from kafka import KafkaProducer
import json


class TransactionProducer:
    def __init__(self, bootstrap_servers="localhost:9092"):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

    def send_transaction(self, topic, transaction_data):
        self.producer.send(topic, transaction_data)
        self.producer.flush()


# Example usage:
# producer = TransactionProducer()
# producer.send_transaction('fraud-transactions', {'amount': 500, 'ip': '192.168.1.1'})
