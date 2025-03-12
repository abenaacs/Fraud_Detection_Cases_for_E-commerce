# Fraud Detection Project

This repository contains the code and documentation for improving fraud detection in e-commerce and banking transactions. The project focuses on preprocessing transaction data, engineering features, building machine learning models, explaining predictions using SHAP and LIME, deploying the solution as a Flask API for real-time fraud detection, and visualizing insights through an interactive Dash dashboard.

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Dependencies](#dependencies)
4. [Setup Instructions](#setup-instructions)
5. [How to Run](#how-to-run)
6. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
7. [Model Building and Training](#model-building-and-training)
8. [Model Explainability](#model-explainability)
9. [Model Deployment and API Development](#model-deployment-and-api-development)
10. [Dashboard Development](#dashboard-development)
11. [Running Tests](#running-tests)
12. [Contributing](#contributing)
13. [License](#license)

---

## Overview

The goal of this project is to create accurate and robust fraud detection models for e-commerce and banking transactions. The solution leverages geolocation analysis, transaction pattern recognition, advanced machine learning techniques, and real-time deployment to identify fraudulent activities.
Key tasks include:

- Preprocessing and cleaning transaction data.
- Engineering features such as time-based features and geolocation mapping.
- Building and training machine learning models.
- Explaining model predictions using SHAP and LIME.
- Deploying the trained model as a Flask API for real-time fraud detection.
- Visualizing fraud insights using an interactive Dash dashboard.

## Directory Structure

```
fraud_detection/
│
├── data/                     # Raw and processed datasets
│   ├── Fraud_Data.csv        # E-commerce transaction data
│   ├── IpAddress_to_Country.csv  # IP address to country mapping
│   └── creditcard.csv        # Bank transaction data
│
├── models/                   # Trained machine learning models
│   └── fraud_detection_model.pkl
│
├── src/                      # Source code for data processing, modeling, and explainability
│   ├── __init__.py           # Package initialization
│   ├── data_loader.py        # Module for loading datasets
│   ├── data_cleaner.py       # Module for cleaning and preprocessing data
│   ├── feature_engineer.py   # Module for feature engineering
│   ├── model_builder.py      # Module for model building and evaluation
│   ├── mlflow_utils.py       # Module for MLflow integration
│   ├── explainability.py     # Module for SHAP and LIME explainability
│   ├── serve_model.py        # Flask API for serving the model
│   └── dashboard.py          # Dash dashboard for visualizing fraud insights
│
├── tests/                    # Unit tests for modules
│   ├── test_data_loader.py   # Tests for data loader
│   ├── test_data_cleaner.py  # Tests for data cleaner
│   ├── test_feature_engineer.py  # Tests for feature engineer
│   ├── test_model_builder.py # Tests for model builder
│   └── test_explainability.py # Tests for explainability
│
├── notebooks/                # Jupyter Notebooks for exploratory data analysis
│   └── exploratory_data_analysis.ipynb  # Notebook for EDA
│
├── Dockerfile                # Docker configuration for Flask API
├── requirements.txt          # List of Python dependencies
└── README.md                 # This file
```

## Dependencies

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Required Libraries

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical computations.
- `matplotlib` and `seaborn`: For data visualization.
- `scikit-learn`: For machine learning algorithms and evaluation metrics.
- `mlflow`: For experiment tracking and model versioning.
- `shap` and `lime`: For model explainability.
- `flask`: For serving the trained model as an API.
- `dash`: For building interactive dashboards.
- `unittest`: For unit testing.

### Resolving Dependency Conflicts

The project includes `faust` and `flask`, which have conflicting dependencies on the `click` library. To resolve this conflict, you can either downgrade `flask` to a version compatible with `click<8.0` or replace `faust` with an alternative library.

#### Option 1: Downgrade `flask`

Update your `requirements.txt` to use a compatible version of `flask`:

```plaintext
faust==1.10.4
flask>=2.3.0,<3.0.0
mlflow>=2.10.0
```

Then run:

```bash
pip install -r requirements.txt
```

#### Option 2: Remove or Replace `faust`

If `faust` is not critical, you can remove it from your `requirements.txt`:

```plaintext
flask>=3.0.0
mlflow>=2.10.0
```

Then run:

```bash
pip install -r requirements.txt
```

## Setup Instructions

1. **Clone the Repository**:

```bash
git clone https://github.com/abenaacs/Fraud_Detection_Cases_for_E-commerce.git
cd Fraud_Detection_Cases_for_E-commerce
```

2. **Install Dependencies**:

```bash
pip install -r requirements.txt
```

3. **Prepare the Data**:
   Place your raw datasets (`Fraud_Data.csv`, `IpAddress_to_Country.csv`, and `creditcard.csv`) in the `data/` folder.
4. **Generate Folder Structure** (if not already created):
   Run the `folder.py` script to generate the folder structure:

```bash
python folder.py
```

---

## How to Run

### Step 1: Preprocess the Data

Run the preprocessing pipeline to clean and process the data:

```bash
python src/main.py
```

This will:

- Load the datasets.
- Clean and preprocess the data.
- Engineer new features (e.g., time-based features, geolocation mapping).
- Save the processed data in the `data/` folder as `processed_fraud_data.csv` and `processed_creditcard_data.csv`.

### Step 2: Perform Exploratory Data Analysis (EDA)

Open the Jupyter Notebook for EDA:

```bash
jupyter notebook notebooks/exploratory_data_analysis.ipynb
```

The notebook contains:

- Univariate and bivariate analysis of the data.
- Visualizations to understand patterns and relationships in the data.

### Step 3: Train Models

The `main.py` script automatically trains multiple models (e.g., Logistic Regression, Random Forest) and evaluates their performance. Results are printed to the console.

### Step 4: Generate Model Explainability

SHAP and LIME explanations are generated for the best-performing models. Visualizations are saved in the root directory:

- SHAP plots: `shap_summary_plot.png`, `shap_force_plot.png`, `shap_dependence_plot.png`
- LIME explanation: `lime_explanation.html`

### Step 5: Track Experiments with MLflow

Start the MLflow UI to view experiment results:

```bash
mlflow ui
```

Access the dashboard at `http://localhost:5000`.

### Step 6: Deploy the Flask API

1. Start the Flask API:

```bash
python src/serve_model.py
```

The API will be available at `http://localhost:5000`. 2. Test the `/predict` endpoint:
Use `curl` or Postman to send a POST request:

```bash
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{"features": [0.1, 0.2, 0.3, 0.4, 0.5]}'
```

3. Dockerize the API:
   Build and run the Docker container:

```bash
docker build -t fraud-detection-api .
docker run -p 5000:5000 fraud-detection-api
```

### Step 7: Launch the Dash Dashboard

1. Start the Dash app:

```bash
python src/dashboard.py
```

The dashboard will be available at `http://localhost:8050`. 2. Explore the dashboard:

- Summary boxes display total transactions, fraud cases, and fraud percentages.
- Line charts show fraud trends over time.
- Bar charts analyze fraud cases by device, browser, and geography.

## Exploratory Data Analysis (EDA)

The `exploratory_data_analysis.ipynb` notebook provides insights into the dataset. Key analyses include:

- Distribution of fraudulent vs. non-fraudulent transactions.
- Correlation between features.
- Patterns in transaction amounts, times, and geolocations.

## Model Building and Training

The `model_builder.py` script handles model selection, training, and evaluation. It includes:

- Multiple algorithms: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, MLP.
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
- Experiment tracking with MLflow.

## Model Explainability

The `explainability.py` script generates SHAP and LIME explanations for the best-performing models:

- SHAP provides global and local interpretability (summary plot, force plot, dependence plot).
- LIME explains individual predictions using interpretable surrogate models.
  Explainability outputs are saved in the root directory:
- SHAP visualizations: `shap_summary_plot.png`, `shap_force_plot.png`, `shap_dependence_plot.png`.
- LIME explanation: `lime_explanation.html`.

## Model Deployment and API Development

The `serve_model.py` script serves the trained model using Flask. Key features include:

- A `/predict` endpoint for real-time fraud detection.
- Logging to track incoming requests, errors, and fraud predictions.
- Dockerized deployment for scalability and portability.

## Dashboard Development

The `dashboard.py` script creates an interactive dashboard using Dash. Key features include:

- Summary boxes displaying total transactions, fraud cases, and fraud percentages.
- Line charts showing fraud trends over time.
- Bar charts analyzing fraud cases by device, browser, and geography.

---

## Running Tests

The project includes unit tests for all modules. To run all tests:

```bash
python -m unittest discover -s tests
```

Alternatively, you can run individual test files:

```bash
python tests/test_data_loader.py
python tests/test_data_cleaner.py
python tests/test_feature_engineer.py
python tests/test_model_builder.py
python tests/test_explainability.py
```

## Contributing

We welcome contributions to improve the project! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:

```bash
git checkout -b feature-name
```

3. Commit your changes:

```bash
git commit -m "Add feature or fix"
```

4. Push your changes to GitHub:

```bash
git push origin feature-name
```

5. Open a pull request describing your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to 10X Academy for providing the datasets and business context.
- Special thanks to the open-source community for tools like Pandas, NumPy, Matplotlib, Scikit-learn, SHAP, and LIME.
