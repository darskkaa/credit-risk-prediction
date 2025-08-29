"""
Configuration file for Credit Default Risk Prediction Model
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# FMP API Configuration
FMP_API_KEY = os.getenv('FMP_API_KEY')
FMP_BASE_URL = 'https://financialmodelingprep.com/api/v3'

# API Endpoints
ENDPOINTS = {
    'company_list': '/available-traded',
    'income_statement': '/income-statement',
    'balance_sheet': '/balance-sheet-statement',
    'cash_flow': '/cash-flow-statement',
    'ratios': '/ratios',
    'company_profile': '/profile'
}

# Data Collection Settings
MAX_COMPANIES = 20  # Conservative limit for free tier (250 requests/day)
REQUEST_DELAY = 1.0  # Delay between API requests in seconds
DATA_DIR = 'data'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = 'models'

# Financial Statement Periods
PERIODS = ['annual', 'quarterly']
MAX_PERIODS = 5  # Number of years/quarters to fetch

# Feature Engineering Settings
FEATURE_GROUPS = {
    'liquidity': ['currentRatio', 'quickRatio', 'cashRatio'],
    'solvency': ['debtToEquity', 'debtToAssets', 'interestCoverage'],
    'profitability': ['returnOnAssets', 'returnOnEquity', 'grossProfitMargin'],
    'efficiency': ['assetTurnover', 'inventoryTurnover', 'receivablesTurnover'],
    'growth': ['revenueGrowth', 'assetGrowth', 'equityGrowth']
}

# Default Event Definition
DEFAULT_CRITERIA = {
    'bankruptcy_filing': True,
    'delisting': True,
    'credit_rating_downgrade': 'CCC'  # Below investment grade
}

# Machine Learning Settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Model Parameters
MODEL_PARAMS = {
    'logistic_regression': {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    },
    'random_forest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'xgboost': {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }
}

# Performance Metrics
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Flask API Settings
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = False

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = 'credit_default_prediction.log'

# Data Validation
MIN_FEATURES = 10  # Minimum number of features required
MIN_SAMPLES = 50   # Minimum number of samples required
MAX_MISSING_RATIO = 0.3  # Maximum ratio of missing values allowed
