"""
Data Preprocessing Module for Credit Default Risk Prediction Model

This module handles data cleaning, feature engineering, and preparation
for machine learning model training.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import config
from utils import (
    setup_logging, create_directories, save_data, load_data,
    calculate_financial_ratios, handle_missing_values, validate_data_quality
)

class DataPreprocessor:
    """Handles data preprocessing and feature engineering"""
    
    def __init__(self):
        self.logger = setup_logging(__name__)
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoder = LabelEncoder()
        create_directories()
        
    def load_raw_data(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """
        Load raw data for a specific company
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            Dictionary containing all raw data for the company
        """
        self.logger.info(f"Loading raw data for {ticker}")
        
        company_data = {}
        
        for period in config.PERIODS:
            # Load financial statements
            for statement_type in ['income', 'balance', 'cash_flow']:
                filename = f'{ticker}_{statement_type}_{period}.csv'
                filepath = os.path.join(config.RAW_DATA_DIR, filename)
                
                if os.path.exists(filepath):
                    try:
                        df = load_data(filename, config.RAW_DATA_DIR)
                        company_data[f'{statement_type}_{period}'] = df
                    except Exception as e:
                        self.logger.warning(f"Failed to load {filename}: {e}")
            
            # Load financial ratios
            ratios_filename = f'{ticker}_ratios_{period}.csv'
            ratios_filepath = os.path.join(config.RAW_DATA_DIR, ratios_filename)
            
            if os.path.exists(ratios_filepath):
                try:
                    ratios_df = load_data(ratios_filename, config.RAW_DATA_DIR)
                    company_data[f'ratios_{period}'] = ratios_df
                except Exception as e:
                    self.logger.warning(f"Failed to load {ratios_filename}: {e}")
        
        self.logger.info(f"Loaded {len(company_data)} datasets for {ticker}")
        return company_data
    
    def merge_financial_data(self, company_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge all financial data for a company into a single DataFrame
        
        Args:
            company_data: Dictionary containing company data
            
        Returns:
            Merged DataFrame with all financial data
        """
        self.logger.info("Merging financial data...")
        
        merged_data = []
        
        # Process each period separately
        for period in config.PERIODS:
            period_data = {}
            
            # Get income statement
            income_key = f'income_{period}'
            if income_key in company_data:
                income_df = company_data[income_key].copy()
                for col in income_df.columns:
                    if col not in ['date', 'ticker', 'statement_type', 'period']:
                        period_data[f'inc_{col}'] = income_df[col].iloc[0] if not income_df.empty else np.nan
            
            # Get balance sheet
            balance_key = f'balance_{period}'
            if balance_key in company_data:
                balance_df = company_data[balance_key].copy()
                for col in balance_df.columns:
                    if col not in ['date', 'ticker', 'statement_type', 'period']:
                        period_data[f'bs_{col}'] = balance_df[col].iloc[0] if not balance_df.empty else np.nan
            
            # Get cash flow statement
            cash_flow_key = f'cash_flow_{period}'
            if cash_flow_key in company_data:
                cash_flow_df = company_data[cash_flow_key].copy()
                for col in cash_flow_df.columns:
                    if col not in ['date', 'ticker', 'statement_type', 'period']:
                        period_data[f'cf_{col}'] = cash_flow_df[col].iloc[0] if not cash_flow_df.empty else np.nan
            
            # Get financial ratios
            ratios_key = f'ratios_{period}'
            if ratios_key in company_data:
                ratios_df = company_data[ratios_key].copy()
                for col in ratios_df.columns:
                    if col not in ['date', 'ticker', 'period']:
                        period_data[f'ratio_{col}'] = ratios_df[col].iloc[0] if not ratios_df.empty else np.nan
            
            # Add metadata
            if any(period_data):
                period_data['period'] = period
                period_data['ticker'] = company_data.get('ticker', '')
                merged_data.append(period_data)
        
        if not merged_data:
            self.logger.warning("No data to merge")
            return pd.DataFrame()
        
        merged_df = pd.DataFrame(merged_data)
        self.logger.info(f"Merged data shape: {merged_df.shape}")
        
        return merged_df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional engineered features from raw financial data
        
        Args:
            df: Input DataFrame with financial data
            
        Returns:
            DataFrame with additional engineered features
        """
        self.logger.info("Engineering additional features...")
        
        df_engineered = df.copy()
        
        # Calculate additional financial ratios if raw data is available
        if 'bs_totalAssets' in df.columns and 'bs_totalLiabilities' in df.columns:
            df_engineered['debt_to_assets'] = df['bs_totalLiabilities'] / df['bs_totalAssets']
        
        if 'bs_totalStockholdersEquity' in df.columns and 'bs_totalAssets' in df.columns:
            df_engineered['equity_multiplier'] = df['bs_totalAssets'] / df['bs_totalStockholdersEquity']
        
        if 'inc_revenue' in df.columns and 'bs_totalAssets' in df.columns:
            df_engineered['asset_turnover'] = df['inc_revenue'] / df['bs_totalAssets']
        
        if 'inc_netIncome' in df.columns and 'bs_totalStockholdersEquity' in df.columns:
            df_engineered['roe'] = df['inc_netIncome'] / df['bs_totalStockholdersEquity']
        
        if 'inc_netIncome' in df.columns and 'bs_totalAssets' in df.columns:
            df_engineered['roa'] = df['inc_netIncome'] / df['bs_totalAssets']
        
        # Calculate growth rates (year-over-year)
        numeric_columns = df_engineered.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col not in ['period', 'ticker']:
                # Calculate growth rate
                df_engineered[f'{col}_growth'] = df_engineered[col].pct_change()
                
                # Calculate rolling averages
                df_engineered[f'{col}_rolling_mean'] = df_engineered[col].rolling(window=2, min_periods=1).mean()
                df_engineered[f'{col}_rolling_std'] = df_engineered[col].rolling(window=2, min_periods=1).std()
        
        self.logger.info(f"Engineered features added. New shape: {df_engineered.shape}")
        return df_engineered
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and outliers
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Cleaning data...")
        
        # Remove rows with too many missing values
        missing_threshold = len(df.columns) * 0.5
        df_cleaned = df.dropna(thresh=missing_threshold)
        
        # Handle remaining missing values
        df_cleaned = handle_missing_values(df_cleaned, strategy='median')
        
        # Remove infinite values
        df_cleaned = df_cleaned.replace([np.inf, -np.inf], np.nan)
        df_cleaned = df_cleaned.dropna()
        
        # Remove duplicate rows
        df_cleaned = df_cleaned.drop_duplicates()
        
        self.logger.info(f"Data cleaning completed. Shape: {df_cleaned.shape}")
        return df_cleaned
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for machine learning
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features DataFrame, feature names)
        """
        self.logger.info("Preparing features for ML...")
        
        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove non-feature columns
        exclude_cols = ['period', 'ticker']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Create features DataFrame
        features_df = df[feature_cols].copy()
        
        # Remove columns with too many missing values
        missing_ratio = features_df.isnull().sum() / len(features_df)
        valid_features = missing_ratio[missing_ratio < 0.3].index.tolist()
        
        features_df = features_df[valid_features]
        
        self.logger.info(f"Prepared {len(valid_features)} features")
        return features_df, valid_features
    
    def create_default_labels(self, df: pd.DataFrame, 
                            default_criteria: Dict = None) -> pd.Series:
        """
        Create default labels based on financial criteria
        
        Args:
            df: Input DataFrame
            default_criteria: Criteria for defining defaults
            
        Returns:
            Series with default labels (1 for default, 0 for non-default)
        """
        self.logger.info("Creating default labels...")
        
        if default_criteria is None:
            default_criteria = config.DEFAULT_CRITERIA
        
        # Initialize labels as non-default (0)
        labels = pd.Series(0, index=df.index)
        
        # Apply default criteria
        for idx, row in df.iterrows():
            is_default = False
            
            # Check debt-to-equity ratio
            if 'ratio_debtToEquity' in row and pd.notna(row['ratio_debtToEquity']):
                if row['ratio_debtToEquity'] > 2.0:  # High debt ratio
                    is_default = True
            
            # Check interest coverage ratio
            if 'ratio_interestCoverage' in row and pd.notna(row['ratio_interestCoverage']):
                if row['ratio_interestCoverage'] < 1.5:  # Low interest coverage
                    is_default = True
            
            # Check current ratio
            if 'ratio_currentRatio' in row and pd.notna(row['ratio_currentRatio']):
                if row['ratio_currentRatio'] < 1.0:  # Low liquidity
                    is_default = True
            
            # Check return on assets
            if 'ratio_returnOnAssets' in row and pd.notna(row['ratio_returnOnAssets']):
                if row['ratio_returnOnAssets'] < -0.1:  # Negative ROA
                    is_default = True
            
            if is_default:
                labels[idx] = 1
        
        default_count = labels.sum()
        total_count = len(labels)
        
        self.logger.info(f"Default labels created: {default_count} defaults out of {total_count} samples ({default_count/total_count:.2%})")
        
        return labels
    
    def scale_features(self, features_df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale features using StandardScaler
        
        Args:
            features_df: Features DataFrame
            fit: Whether to fit the scaler (True for training, False for prediction)
            
        Returns:
            Scaled features DataFrame
        """
        self.logger.info("Scaling features...")
        
        if fit:
            scaled_features = self.scaler.fit_transform(features_df)
        else:
            scaled_features = self.scaler.transform(features_df)
        
        scaled_df = pd.DataFrame(scaled_features, columns=features_df.columns, index=features_df.index)
        
        self.logger.info("Features scaled successfully")
        return scaled_df
    
    def handle_imbalanced_data(self, features_df: pd.DataFrame, labels: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle imbalanced data using SMOTE
        
        Args:
            features_df: Features DataFrame
            labels: Target labels
            
        Returns:
            Tuple of (balanced features, balanced labels)
        """
        self.logger.info("Handling imbalanced data with SMOTE...")
        
        # Check if SMOTE is needed
        class_counts = labels.value_counts()
        if len(class_counts) < 2:
            self.logger.warning("Only one class found. Cannot apply SMOTE.")
            return features_df, labels
        
        if class_counts.min() / class_counts.max() > 0.3:
            self.logger.info("Data is not significantly imbalanced. Skipping SMOTE.")
            return features_df, labels
        
        try:
            smote = SMOTE(random_state=config.RANDOM_STATE)
            features_balanced, labels_balanced = smote.fit_resample(features_df, labels)
            
            self.logger.info(f"SMOTE applied. New shape: {features_balanced.shape}")
            self.logger.info(f"Class distribution: {pd.Series(labels_balanced).value_counts().to_dict()}")
            
            return features_balanced, labels_balanced
            
        except Exception as e:
            self.logger.warning(f"SMOTE failed: {e}. Using original data.")
            return features_df, labels
    
    def preprocess_company_data(self, ticker: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Complete preprocessing pipeline for a single company
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            Tuple of (features, labels, feature_names)
        """
        self.logger.info(f"Preprocessing data for {ticker}")
        
        # Load raw data
        company_data = self.load_raw_data(ticker)
        
        if not company_data:
            self.logger.warning(f"No data found for {ticker}")
            return pd.DataFrame(), pd.Series(), []
        
        # Merge data
        merged_df = self.merge_financial_data(company_data)
        
        if merged_df.empty:
            self.logger.warning(f"No data to merge for {ticker}")
            return pd.DataFrame(), pd.Series(), []
        
        # Engineer features
        engineered_df = self.engineer_features(merged_df)
        
        # Clean data
        cleaned_df = self.clean_data(engineered_df)
        
        # Create labels
        labels = self.create_default_labels(cleaned_df)
        
        # Prepare features
        features_df, feature_names = self.prepare_features(cleaned_df)
        
        if features_df.empty:
            self.logger.warning(f"No features available for {ticker}")
            return pd.DataFrame(), pd.Series(), []
        
        # Scale features
        scaled_features = self.scale_features(features_df)
        
        # Handle imbalanced data
        balanced_features, balanced_labels = self.handle_imbalanced_data(scaled_features, labels)
        
        return balanced_features, balanced_labels, feature_names
    
    def preprocess_all_data(self, company_list: List[str]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Preprocess data for all companies
        
        Args:
            company_list: List of company tickers
            
        Returns:
            Tuple of (all features, all labels, feature names)
        """
        self.logger.info(f"Preprocessing data for {len(company_list)} companies")
        
        all_features = []
        all_labels = []
        feature_names = []
        
        for ticker in company_list:
            try:
                features, labels, names = self.preprocess_company_data(ticker)
                
                if not features.empty and not labels.empty:
                    all_features.append(features)
                    all_labels.append(labels)
                    
                    if not feature_names:
                        feature_names = names
                    
                    self.logger.info(f"Successfully preprocessed {ticker}")
                else:
                    self.logger.warning(f"Failed to preprocess {ticker}")
                    
            except Exception as e:
                self.logger.error(f"Error preprocessing {ticker}: {e}")
                continue
        
        if not all_features:
            self.logger.error("No data was successfully preprocessed")
            return pd.DataFrame(), pd.Series(), []
        
        # Combine all data
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_labels = pd.concat(all_labels, ignore_index=True)
        
        self.logger.info(f"Combined data shape: {combined_features.shape}")
        
        # Save preprocessed data
        save_data(combined_features, 'preprocessed_features.csv')
        save_data(combined_labels.to_frame('default'), 'preprocessed_labels.csv')
        
        return combined_features, combined_labels, feature_names

def main():
    """Main function for data preprocessing"""
    try:
        # Load company list
        company_list_path = os.path.join(config.RAW_DATA_DIR, 'company_list.csv')
        
        if not os.path.exists(company_list_path):
            print("Company list not found. Please run data_collection.py first.")
            return
        
        company_list_df = pd.read_csv(company_list_path)
        company_list = company_list_df['symbol'].tolist()
        
        # Limit companies for processing
        company_list = company_list[:min(len(company_list), 50)]  # Process first 50 companies
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Preprocess all data
        features, labels, feature_names = preprocessor.preprocess_all_data(company_list)
        
        if not features.empty:
            print(f"\nData preprocessing completed successfully!")
            print(f"Final dataset shape: {features.shape}")
            print(f"Number of features: {len(feature_names)}")
            print(f"Class distribution: {labels.value_counts().to_dict()}")
            print(f"Preprocessed data saved to: {config.PROCESSED_DATA_DIR}")
        else:
            print("Data preprocessing failed. Check logs for details.")
        
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        raise

if __name__ == "__main__":
    main()
