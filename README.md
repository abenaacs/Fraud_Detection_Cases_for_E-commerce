# Fraud Detection Project

This repository contains the code and documentation for improving fraud detection in e-commerce and banking transactions. The project focuses on preprocessing transaction data, engineering features, and building machine learning models to detect fraudulent activities.

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Dependencies](#dependencies)
4. [Setup Instructions](#setup-instructions)
5. [How to Run](#how-to-run)
6. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
7. [Running Tests](#running-tests)
8. [Contributing](#contributing)
9. [License](#license)

---

## Overview

The goal of this project is to create accurate and robust fraud detection models for e-commerce and banking transactions. The solution leverages geolocation analysis, transaction pattern recognition, and advanced machine learning techniques to identify fraudulent activities.

Key tasks include:

- Preprocessing and cleaning transaction data.
- Engineering features such as time-based features and geolocation mapping.
- Building and training machine learning models.
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
├── src/                      # Source code for data processing and feature engineering
│   ├── __init__.py           # Package initialization
│   ├── data_loader.py        # Module for loading datasets
│   ├── data_cleaner.py       # Module for cleaning and preprocessing data
│   ├── feature_engineer.py   # Module for feature engineering
│   └── main.py               # Main script to execute preprocessing pipeline
│
├── tests/                    # Unit tests for modules
│   ├── test_data_loader.py   # Tests for data loader
│   ├── test_data_cleaner.py  # Tests for data cleaner
│   └── test_feature_engineer.py  # Tests for feature engineer
│
├── notebooks/                # Jupyter Notebooks for exploratory data analysis
│   └── exploratory_data_analysis.ipynb  # Notebook for EDA
│
├── requirements.txt          # List of Python dependencies
└── README.md                 # This file
```

---

## Dependencies

The project requires the following Python libraries:

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical computations.
- `matplotlib` and `seaborn`: For data visualization.
- `unittest`: For unit testing.

Install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

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

---

## Exploratory Data Analysis (EDA)

The `exploratory_data_analysis.ipynb` notebook provides insights into the dataset. Key analyses include:

- Distribution of fraudulent vs. non-fraudulent transactions.
- Correlation between features.
- Patterns in transaction amounts, times, and geolocations.

---

## Running Tests

The project includes unit tests for the `src/` modules. To run all tests:

```bash
python -m unittest discover -s tests
```

Alternatively, you can run individual test files:

```bash
python tests/test_data_loader.py
python tests/test_data_cleaner.py
python tests/test_feature_engineer.py
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
- Special thanks to the open-source community for tools like Pandas, NumPy, and Matplotlib.
