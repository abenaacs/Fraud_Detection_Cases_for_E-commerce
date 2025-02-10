import mlflow


def setup_mlflow(experiment_name="Fraud_Detection_Experiment"):
    mlflow.set_experiment(experiment_name)
    print(f"MLflow experiment '{experiment_name}' has been set up.")
