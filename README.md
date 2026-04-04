<div align="center">
<h1>Gold Market Intelligence</h2>
<h3>Adaptive Forecasting & Risk Monitoring System</h3>
</div>
<div align="center">

![Python](https://img.shields.io/badge/python-3.10+-blue)
![MLflow](https://img.shields.io/badge/MLflow-Model%20Registry-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![CI/CD](https://img.shields.io/badge/GitHub%20Actions-CI/CD-black)

Production-ready MLOps system for forecasting gold prices and monitoring market risk using time-series modeling, drift detection, and continual learning.

</div>

## Table of Contents

- [Overview](#overview)
- [Machine Learning Tasks](#machine-learning-tasks)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Codespaces](#running-the-project-with-github-codespaces)
- [Data Ingestion & Preprocessing](#data-ingestion--preprocessing)
- [Tech Stack](#tech-stack)
- [License](#license)

## Overview

Gold Market MLOps is an end-to-end machine learning system designed to forecast gold prices and classify market risk levels within a production-ready MLOps pipeline.

The system integrates data ingestion, feature engineering, model training, evaluation, deployment, monitoring, and automated retraining into a unified workflow.

Key capabilities:

- 7-day ahead gold price forecasting  
- Volatility-based risk classification  
- Drift detection (covariate and concept drift)  
- Sliding-window continual learning  
- Model versioning with MLflow  
- REST API deployment with FastAPI  
- Containerized execution with Docker  
- CI/CD integration using GitHub Actions  

## Machine Learning Tasks

### 1. Time-Series Regression

- Objective: Forecast gold prices 7 days ahead    
- Validation: Time-based split & rolling window backtesting  
- Metrics: MAE, RMSE, MAPE, Mean Directional Accuracy  

### 2. Risk Classification

- Objective: Classify market condition into Low, Medium, High Risk  
- Labeling based on volatility distribution  
- Metric: F1-score (focus on High Risk class)  

## System Architecture

The system consists of four primary layers:

### Data Engineering Layer

- Daily ingestion from financial API (XAU/USD)
- Schema validation and anomaly checking
- Feature engineering (lag features, rolling mean, rolling standard deviation, returns)
- Time-aware processing to prevent data leakage

### Machine Learning Layer

- Rolling window training
- Time-based validation
- Backtesting evaluation
- MLflow model tracking and registry

### Deployment Layer

- Docker containerization
- CI/CD workflow using GitHub Actions

### Monitoring & Continual Learning Layer

- Performance monitoring (MAPE threshold tracking)
- Distribution shift detection
- Drift-triggered retraining
- Scheduled retraining strategy
- Structured inference logging


## Project Structure

```bash
gold-market-mlops/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── monitoring/
│   ├── retraining/
│   └── registry/
│
├── api/
├── configs/
├── notebooks/
├── tests/
└── requirements.txt
```
## Running the Project with GitHub Codespaces

This project can be executed directly in a cloud development environment using **GitHub Codespaces**, allowing users to run the full MLOps pipeline without installing dependencies locally.
1. Go to the repository on GitHub.
2. Click **Code → Codespaces → Create Codespace on main**.
3. Wait until the development environment finishes initializing.

GitHub Codespaces will automatically provide a cloud-based development environment with the necessary tools.

## Data Ingestion & Preprocessing

This project includes an automated data pipeline for collecting and preprocessing financial time-series data to support continual learning.

### 1. Data Ingestion
---
To fetch the latest raw data (gold, oil, and macroeconomic indicators):

```bash
python src/data/ingestion.py
```

This will generate raw datasets in:

```
data/raw/
```

Generated files:

* gold_prices.csv
* oil_prices.csv
* macro_fred.csv

### 2. Data Preprocessing
---
To preprocess each dataset individually:

```bash
python src/data/preprocess_gold.py
python src/data/preprocess_oil.py
python src/data/preprocess_fred.py
```

To merge all processed datasets:

```bash
python src/data/preprocess_merge.py
```

Processed data will be stored in:

```
data/processed/
```

### 3. Run Full Data Pipeline (Recommended)
---
To execute the full pipeline (ingestion → preprocessing → merging):

```bash
python src/data/data_pipeline.py
```

This script will:

1. Fetch latest data
2. Generate features for each dataset
3. Merge all datasets into a unified dataset

Final output:

```
data/processed/market_dataset.csv
```

### Notes
---
- The pipeline can be executed repeatedly to simulate continual learning
- Raw data is stored separately from processed data
- Make sure to set environment variable FRED_API_KEY before running ingestion

## Tech Stack

- Python
- XGBoost / LightGBM
- MLflow
- FastAPI
- Docker
- GitHub Actions
- Pandas / NumPy / Scikit-learn

## Contributors
- Aufii Fathin Nabila

## License

MIT License
