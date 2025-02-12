# Fraud Detection Project

This repository contains the code and documentation for improving fraud detection in e-commerce and banking transactions. The project focuses on preprocessing transaction data, engineering features, building machine learning models, explaining predictions using SHAP and LIME, and deploying the solution for real-time fraud detection.

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Dependencies](#dependencies)
4. [Setup Instructions](#setup-instructions)
5. [How to Run](#how-to-run)
6. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
7. [Model Building and Training](#model-building-and-training)
8. [Model Explainability](#model-explainability)
9. [Running Tests](#running-tests)
10. [Contributing](#contributing)
11. [License](#license)

---

## Overview

The goal of this project is to create accurate and robust fraud detection models for e-commerce and banking transactions. The solution leverages geolocation analysis, transaction pattern recognition, and advanced machine learning techniques to identify fraudulent activities.

Key tasks include:

- Preprocessing and cleaning transaction data.
- Engineering features such as time-based features and geolocation mapping.
- Building and training machine learning models.
- Explaining model predictions using SHAP and LIME.
- Evaluating model performance and deploying it for real-time fraud detection.

---

## Directory Structure

```
fraud_detection/
│
├── data/                     # Raw and processed datasets
│   ├── Fraud_Data.csv        # E-commerce transaction data
│   ├── IpAddress_to_Country.csv  # IP address to country mapping
│   └── creditcard.csv        # Bank transaction data
│
├── src/                      # Source code for data processing, modeling, and explainability
│   ├── __init__.py           # Package initialization
│   ├── data_loader.py        # Module for loading datasets
│   ├── data_cleaner.py       # Module for cleaning and preprocessing data
│   ├── feature_engineer.py   # Module for feature engineering
│   ├── model_builder.py      # Module for model building and evaluation
│   ├── mlflow_utils.py       # Module for MLflow integration
│   ├── explainability.py     # Module for SHAP and LIME explainability
│   └── main.py               # Main script to execute the pipeline
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
├── requirements.txt          # List of Python dependencies
└── README.md                 # This file
```

---

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
- `unittest`: For unit testing.

---

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

---

## Exploratory Data Analysis (EDA)

The `exploratory_data_analysis.ipynb` notebook provides insights into the dataset. Key analyses include:

- Distribution of fraudulent vs. non-fraudulent transactions.
- Correlation between features.
- Patterns in transaction amounts, times, and geolocations.

---

## Model Building and Training

The `model_builder.py` script handles model selection, training, and evaluation. It includes:

- Multiple algorithms: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, MLP.
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
- Experiment tracking with MLflow.

---

## Model Explainability

The `explainability.py` script generates SHAP and LIME explanations for the best-performing models:

- SHAP provides global and local interpretability (summary plot, force plot, dependence plot).
- LIME explains individual predictions using interpretable surrogate models.

Explainability outputs are saved in the root directory:

- SHAP visualizations: `shap_summary_plot.png`, `shap_force_plot.png`, `shap_dependence_plot.png`.
- LIME explanation: `lime_explanation.html`.

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

---

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

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Thanks to Adey Innovations Inc. for providing the datasets and business context.
- Special thanks to the open-source community for tools like Pandas, NumPy, Matplotlib, Scikit-learn, SHAP, and LIME.

---

### Notes

1. **Reproducibility**: Ensure that all dependencies are installed using `requirements.txt` to avoid compatibility issues.
2. **MLflow Dashboard**: Start the MLflow UI to view experiment results:

   ```bash
   mlflow ui
   ```

   Access the dashboard at `http://localhost:5000`.

3. **Explainability Outputs**: SHAP and LIME visualizations are saved in the root directory for easy access.
