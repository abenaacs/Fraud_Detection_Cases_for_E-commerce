import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import mlflow
import mlflow.sklearn


class ModelBuilder:
    def __init__(self, data_path, target_column):
        self.data = pd.read_csv(data_path)
        self.target_column = target_column
        self.X = self.data.drop(columns=[target_column])
        self.y = self.data[target_column]
        self.models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "MLP": MLPClassifier(max_iter=500),
        }

    def split_data(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y,
        )

    def train_and_evaluate(self):
        results = {}
        for model_name, model in self.models.items():
            with mlflow.start_run(run_name=model_name):
                # Train the model
                print("X_train columns:", self.X_train.columns)
                print("X_train data types:\n", self.X_train.dtypes)
                print("Columns with missing values:")
                print(self.X_train.isnull().sum())
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)

                # Evaluate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred)
                recall = recall_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)
                roc_auc = roc_auc_score(self.y_test, y_pred)

                # Log metrics and parameters
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("roc_auc", roc_auc)

                # Log the model
                mlflow.sklearn.log_model(model, f"{model_name}_model")

                # Store results
                results[model_name] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "roc_auc": roc_auc,
                }
        return results
