"""
Utility functions for Credit Default Risk Prediction Model
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import requests
from tqdm import tqdm
import config

def setup_logging(name: str = __name__) -> logging.Logger:
    """Set up logging configuration"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(getattr(logging, config.LOG_LEVEL))
        
        # Create formatter
        formatter = logging.Formatter(config.LOG_FORMAT)
        
        # Create file handler
        os.makedirs('logs', exist_ok=True)
        file_handler = logging.FileHandler(config.LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def create_directories() -> None:
    """Create necessary directories for the project"""
    directories = [
        config.DATA_DIR,
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.MODELS_DIR,
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def safe_api_request(url: str, params: Dict = None, max_retries: int = 3) -> Optional[Dict]:
    """
    Make a safe API request with retry logic and error handling
    
    Args:
        url: API endpoint URL
        params: Query parameters
        max_retries: Maximum number of retry attempts
        
    Returns:
        API response as dictionary or None if failed
    """
    logger = setup_logging(__name__)
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Check if response is valid JSON
            try:
                data = response.json()
                return data
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON response from {url}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"API request failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"API request failed after {max_retries} attempts: {url}")
                return None
    
    return None

def calculate_financial_ratios(income_stmt: pd.DataFrame, 
                              balance_sheet: pd.DataFrame, 
                              cash_flow: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate financial ratios from financial statements
    
    Args:
        income_stmt: Income statement data
        balance_sheet: Balance sheet data
        cash_flow: Cash flow statement data
        
    Returns:
        DataFrame with calculated financial ratios
    """
    ratios = pd.DataFrame()
    
    # Merge data on date
    merged = pd.merge(income_stmt, balance_sheet, on='date', how='inner', suffixes=('_inc', '_bs'))
    merged = pd.merge(merged, cash_flow, on='date', how='inner', suffixes=('', '_cf'))
    
    # Liquidity Ratios
    ratios['currentRatio'] = merged['totalCurrentAssets'] / merged['totalCurrentLiabilities']
    ratios['quickRatio'] = (merged['totalCurrentAssets'] - merged['inventory']) / merged['totalCurrentLiabilities']
    ratios['cashRatio'] = merged['cashAndCashEquivalents'] / merged['totalCurrentLiabilities']
    
    # Solvency Ratios
    ratios['debtToEquity'] = merged['totalDebt'] / merged['totalStockholdersEquity']
    ratios['debtToAssets'] = merged['totalDebt'] / merged['totalAssets']
    ratios['interestCoverage'] = merged['ebitda'] / merged['interestExpense']
    
    # Profitability Ratios
    ratios['returnOnAssets'] = merged['netIncome'] / merged['totalAssets']
    ratios['returnOnEquity'] = merged['netIncome'] / merged['totalStockholdersEquity']
    ratios['grossProfitMargin'] = merged['grossProfit'] / merged['revenue']
    ratios['netProfitMargin'] = merged['netIncome'] / merged['revenue']
    
    # Efficiency Ratios
    ratios['assetTurnover'] = merged['revenue'] / merged['totalAssets']
    ratios['inventoryTurnover'] = merged['costOfRevenue'] / merged['inventory']
    ratios['receivablesTurnover'] = merged['revenue'] / merged['netReceivables']
    
    # Growth Metrics
    ratios['revenueGrowth'] = merged['revenue'].pct_change()
    ratios['assetGrowth'] = merged['totalAssets'].pct_change()
    ratios['equityGrowth'] = merged['totalStockholdersEquity'].pct_change()
    
    # Add date column
    ratios['date'] = merged['date']
    
    return ratios

def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Handle missing values in the dataset
    
    Args:
        df: Input DataFrame
        strategy: Strategy for handling missing values ('mean', 'median', 'drop')
        
    Returns:
        DataFrame with handled missing values
    """
    logger = setup_logging(__name__)
    
    # Calculate missing value statistics
    missing_stats = df.isnull().sum()
    missing_ratio = missing_stats / len(df)
    
    logger.info(f"Missing value statistics:")
    for col, ratio in missing_ratio.items():
        if ratio > 0:
            logger.info(f"  {col}: {ratio:.2%} ({missing_stats[col]} values)")
    
    # Apply strategy
    if strategy == 'drop':
        df_cleaned = df.dropna()
        logger.info(f"Dropped rows with missing values. Remaining: {len(df_cleaned)} rows")
    elif strategy == 'mean':
        df_cleaned = df.fillna(df.mean())
        logger.info("Filled missing values with mean")
    elif strategy == 'median':
        df_cleaned = df.fillna(df.median())
        logger.info("Filled missing values with median")
    else:
        logger.warning(f"Unknown strategy '{strategy}'. Using median.")
        df_cleaned = df.fillna(df.median())
    
    return df_cleaned

def validate_data_quality(df: pd.DataFrame, min_features: int = None, 
                         min_samples: int = None, max_missing_ratio: float = None) -> bool:
    """
    Validate data quality for model training
    
    Args:
        df: Input DataFrame
        min_features: Minimum number of features required
        min_samples: Minimum number of samples required
        max_missing_ratio: Maximum ratio of missing values allowed
        
    Returns:
        True if data quality is acceptable, False otherwise
    """
    logger = setup_logging(__name__)
    
    # Use config defaults if not specified
    min_features = min_features or config.MIN_FEATURES
    min_samples = min_samples or config.MIN_SAMPLES
    max_missing_ratio = max_missing_ratio or config.MAX_MISSING_RATIO
    
    # Check number of features
    if len(df.columns) < min_features:
        logger.error(f"Insufficient features: {len(df.columns)} < {min_features}")
        return False
    
    # Check number of samples
    if len(df) < min_samples:
        logger.error(f"Insufficient samples: {len(df)} < {min_samples}")
        return False
    
    # Check missing values
    missing_ratio = df.isnull().sum().max() / len(df)
    if missing_ratio > max_missing_ratio:
        logger.error(f"Too many missing values: {missing_ratio:.2%} > {max_missing_ratio:.2%}")
        return False
    
    logger.info(f"Data quality validation passed:")
    logger.info(f"  Features: {len(df.columns)}")
    logger.info(f"  Samples: {len(df)}")
    logger.info(f"  Max missing ratio: {missing_ratio:.2%}")
    
    return True

def save_data(df: pd.DataFrame, filename: str, directory: str = None) -> str:
    """
    Save DataFrame to file
    
    Args:
        df: DataFrame to save
        filename: Name of the file
        directory: Directory to save in (uses config if None)
        
    Returns:
        Full path to saved file
    """
    if directory is None:
        directory = config.PROCESSED_DATA_DIR
    
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    
    if filename.endswith('.csv'):
        df.to_csv(filepath, index=False)
    elif filename.endswith('.parquet'):
        df.to_parquet(filepath, index=False)
    elif filename.endswith('.json'):
        df.to_json(filepath, orient='records', indent=2)
    else:
        # Default to CSV
        filepath = filepath + '.csv'
        df.to_csv(filepath, index=False)
    
    logger = setup_logging(__name__)
    logger.info(f"Data saved to: {filepath}")
    
    return filepath

def load_data(filename: str, directory: str = None) -> pd.DataFrame:
    """
    Load DataFrame from file
    
    Args:
        filename: Name of the file to load
        directory: Directory to load from (uses config if None)
        
    Returns:
        Loaded DataFrame
    """
    if directory is None:
        directory = config.PROCESSED_DATA_DIR
    
    filepath = os.path.join(directory, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if filename.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filename.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    elif filename.endswith('.json'):
        df = pd.read_json(filepath, orient='records')
    else:
        raise ValueError(f"Unsupported file format: {filename}")
    
    logger = setup_logging(__name__)
    logger.info(f"Data loaded from: {filepath}")
    
    return df

def get_company_info(ticker: str) -> Dict[str, Any]:
    """
    Get basic company information from FMP API
    
    Args:
        ticker: Company ticker symbol
        
    Returns:
        Dictionary with company information
    """
    url = f"{config.FMP_BASE_URL}{config.ENDPOINTS['company_profile']}/{ticker}"
    params = {'apikey': config.FMP_API_KEY}
    
    data = safe_api_request(url, params)
    
    if data and isinstance(data, list) and len(data) > 0:
        return data[0]
    return {}

def format_currency(value: float, currency: str = 'USD') -> str:
    """Format currency values for display"""
    if pd.isna(value):
        return 'N/A'
    
    if abs(value) >= 1e9:
        return f"{value/1e9:.2f}B {currency}"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.2f}M {currency}"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.2f}K {currency}"
    else:
        return f"{value:.2f} {currency}"

def format_percentage(value: float) -> str:
    """Format percentage values for display"""
    if pd.isna(value):
        return 'N/A'
    return f"{value:.2%}"

def calculate_risk_score(probabilities: np.ndarray, threshold: float = 0.5) -> str:
    """
    Convert probability to risk score
    
    Args:
        probabilities: Array of default probabilities
        threshold: Threshold for high risk classification
        
    Returns:
        Risk score string ('Low', 'Medium', 'High')
    """
    if probabilities >= threshold:
        return 'High'
    elif probabilities >= threshold * 0.5:
        return 'Medium'
    else:
        return 'Low'
