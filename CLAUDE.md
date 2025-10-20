# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a submission repository for the 2025 E.SUN AI Challenge - a binary classification competition focused on detecting fraudulent alert accounts using transaction data. The goal is to predict whether E.SUN bank accounts will become alert accounts in the future based on historical transaction patterns.

Competition Details: https://tbrain.trendmicro.com.tw/Competitions/Details/40

## Environment Setup

This project uses `uv` for Python dependency management with Python 3.12.

**Install dependencies:**
```bash
uv sync
```

**Activate virtual environment:**
```bash
source .venv/bin/activate
```

## Data Acquisition

**Download and prepare dataset:**
```bash
bash get_dataset.sh
```

This script:
1. Downloads competition data from Google Drive using `gdown`
2. Extracts CSV files with UTF-8 encoding to `data/` directory
3. Cleans up intermediate directories and zip files

The script handles Chinese filenames correctly and automatically overwrites existing files.

**Expected data files in `data/`:**
- `acct_transaction.csv` - Transaction records (~4M rows)
- `acct_alert.csv` - Alert account labels (~1K rows)
- `acct_predict.csv` - Accounts to predict (~4K rows)
- `submission_template.csv` - Submission format template

## Running the Baseline Model

**Execute the baseline classifier:**
```bash
python TransactionAlertClassifier.py
```

The script is configured to read data from the `data/` directory.

**Output:** `result.csv` - Predictions ready for TBrain submission

## Architecture

### TransactionAlertClassifier.py

This baseline implementation demonstrates the full ML pipeline:

**Pipeline stages:**
1. `LoadCSV(dir_path)` - Loads the 3 competition datasets
2. `PreProcessing(df)` - Feature engineering from transaction data
3. `TrainTestSplit(df, df_alert, df_test)` - Prepares train/test splits with labels
4. `Modeling(X_train, y_train, X_test)` - Trains DecisionTreeClassifier
5. `OutputCSV(path, df_test, X_test, y_pred)` - Generates submission file

**Feature extraction (current baseline):**
- Transaction amount statistics per account (total, max, min, avg for both send/receive)
- Account type classification (E.SUN vs. non-E.SUN)

**Important notes:**
- Training uses only E.SUN accounts (`is_esun==1`) since the prediction set contains only E.SUN accounts
- The baseline intentionally uses a simple approach - participants should improve feature engineering and modeling
- No proper train/validation split is implemented in the baseline

### main.py

Currently a placeholder entry point. Can be used for:
- Orchestrating multiple experiments
- Hyperparameter tuning workflows
- Ensemble model pipelines

## Key Competition Constraints

1. **Training data filtering:** Only E.SUN accounts should be used for training to match the prediction target distribution
2. **Prediction target:** Binary classification (0=normal, 1=alert account)
3. **Temporal nature:** Predict future alert status based on past N months of transactions
4. **Class imbalance:** Alert accounts are rare (~1K alerts vs ~4M transactions)

## Development Workflow

When developing new features or models:

1. **Feature engineering:** Modify `PreProcessing()` to extract new features from transaction data
2. **Model experimentation:** Replace DecisionTreeClassifier in `Modeling()` with more sophisticated approaches
3. **Validation:** Implement proper train/validation split in `TrainTestSplit()` for model evaluation
4. **Submission:** Ensure output format matches `submission_template.csv` structure (acct, label columns)
