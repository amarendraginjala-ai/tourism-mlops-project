# Tourism Package Purchase Prediction — MLOps Project

## Project Overview

This project develops an end-to-end machine learning and MLOps pipeline to predict whether a customer will purchase a tourism package.

The project covers data registration, data preparation, model development, experiment tracking, model registration, automated MLOps workflows, and application deployment.

## Project Architecture

The project follows this workflow:

Dataset
→ Data Preparation
→ Model Training
→ Experiment Tracking
→ Model Registration
→ GitHub Actions
→ Streamlit Deployment

## Dataset

The tourism customer dataset contains customer demographic information and customer interaction attributes.

The target variable is:

`ProdTaken`

where:

- `0` = Customer did not purchase the package
- `1` = Customer purchased the package

The prepared training and testing datasets are registered on the Hugging Face Dataset Hub.

## Data Preparation

The data preparation process includes:

- Removing unnecessary identifier columns
- Handling inconsistent categorical values
- Encoding binary categorical variables
- One-hot encoding categorical variables
- Stratified train-test splitting
- Saving and registering the prepared datasets

The final dataset contains 28 predictor variables.

## Model Development

Three ensemble classification models were evaluated:

- Random Forest
- AdaBoost
- Gradient Boosting

Hyperparameter tuning was performed using cross-validation.

The final registered model is a:

**Random Forest Classifier**

Selected parameters:

- `n_estimators = 200`
- `max_depth = None`
- `min_samples_split = 2`
- `min_samples_leaf = 1`
- `class_weight = balanced`
- `random_state = 42`

## Model Evaluation

The final Random Forest model was evaluated on the test dataset.

| Metric | Score |
|---|---:|
| Accuracy | 0.9080 |
| Precision | 0.9462 |
| Recall | 0.5535 |
| F1 Score | 0.6984 |
| ROC-AUC | 0.9681 |

## Experiment Tracking

MLflow was used during model development to track experiments, parameters, and evaluation metrics.

The local MLflow tracking files are maintained within the project and are excluded from Git version control.

## Hugging Face Model

The trained Random Forest model is registered on the Hugging Face Model Hub.

Model:

`Amarendraa/Tourism-Package-Purchase-RandomForest`

## MLOps Pipeline

GitHub Actions automates the major stages of the project.

The workflow includes:

1. Dataset registration
2. Data preparation
3. Model training
4. Model evaluation
5. Model registration
6. Deployment validation

The workflow is triggered automatically when changes are pushed to the `main` branch.

## Deployment

A Streamlit application provides an interactive interface for package purchase prediction.

The application:

- Accepts customer information
- Applies the required preprocessing
- Loads the trained Random Forest model from Hugging Face
- Generates a purchase prediction
- Displays the estimated purchase probability

## Project Structure

```text
tourism_project/
├── data/
│   ├── tourism.csv
│   ├── train.csv
│   └── test.csv
│
├── deployment/
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── model_building/
│   └── random_forest_tourism_model.joblib
│
├── .github/
│   └── workflows/
│       ├── pipeline.yml
│       └── requirements.txt
│
└── .gitignore
