"""
Data Collection Module for Credit Default Risk Prediction Model

This module handles fetching financial data from the Financial Modeling Prep (FMP) API
including company lists, financial statements, and financial ratios.
"""

import os
import time
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import config
from utils import setup_logging, create_directories, safe_api_request, save_data

class FMPDataCollector:
    """Collects financial data from FMP API"""
    
    def __init__(self):
        self.logger = setup_logging(__name__)
        self.api_key = config.FMP_API_KEY
        
        if not self.api_key:
            raise ValueError("FMP_API_KEY not found in environment variables")
        
        self.base_url = config.FMP_BASE_URL
        create_directories()
        
    def get_company_list(self, limit: int = None) -> List[Dict]:
        """
        Get list of companies using predefined list (fallback for free tier)
        
        Args:
            limit: Maximum number of companies to fetch
            
        Returns:
            List of company dictionaries
        """
        self.logger.info("Using predefined company list (free tier compatible)...")
        
        # Predefined list of well-known companies that should work with free tier
        predefined_companies = [
            {'symbol': 'AAPL', 'name': 'Apple Inc.', 'exchange': 'NASDAQ', 'exchangeShortName': 'NASDAQ', 'type': 'stock'},
            {'symbol': 'MSFT', 'name': 'Microsoft Corporation', 'exchange': 'NASDAQ', 'exchangeShortName': 'NASDAQ', 'type': 'stock'},
            {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'exchange': 'NASDAQ', 'exchangeShortName': 'NASDAQ', 'type': 'stock'},
            {'symbol': 'AMZN', 'name': 'Amazon.com Inc.', 'exchange': 'NASDAQ', 'exchangeShortName': 'NASDAQ', 'type': 'stock'},
            {'symbol': 'TSLA', 'name': 'Tesla Inc.', 'exchange': 'NASDAQ', 'exchangeShortName': 'NASDAQ', 'type': 'stock'},
            {'symbol': 'META', 'name': 'Meta Platforms Inc.', 'exchange': 'NASDAQ', 'exchangeShortName': 'NASDAQ', 'type': 'stock'},
            {'symbol': 'NVDA', 'name': 'NVIDIA Corporation', 'exchange': 'NASDAQ', 'exchangeShortName': 'NASDAQ', 'type': 'stock'},
            {'symbol': 'JPM', 'name': 'JPMorgan Chase & Co.', 'exchange': 'NYSE', 'exchangeShortName': 'NYSE', 'type': 'stock'},
            {'symbol': 'JNJ', 'name': 'Johnson & Johnson', 'exchange': 'NYSE', 'exchangeShortName': 'NYSE', 'type': 'stock'},
            {'symbol': 'V', 'name': 'Visa Inc.', 'exchange': 'NYSE', 'exchangeShortName': 'NYSE', 'type': 'stock'},
            {'symbol': 'PG', 'name': 'Procter & Gamble Co.', 'exchange': 'NYSE', 'exchangeShortName': 'NYSE', 'type': 'stock'},
            {'symbol': 'HD', 'name': 'Home Depot Inc.', 'exchange': 'NYSE', 'exchangeShortName': 'NYSE', 'type': 'stock'},
            {'symbol': 'MA', 'name': 'Mastercard Inc.', 'exchange': 'NYSE', 'exchangeShortName': 'NYSE', 'type': 'stock'},
            {'symbol': 'BAC', 'name': 'Bank of America Corp.', 'exchange': 'NYSE', 'exchangeShortName': 'NYSE', 'type': 'stock'},
            {'symbol': 'DIS', 'name': 'Walt Disney Co.', 'exchange': 'NYSE', 'exchangeShortName': 'NYSE', 'type': 'stock'},
            {'symbol': 'ADBE', 'name': 'Adobe Inc.', 'exchange': 'NASDAQ', 'exchangeShortName': 'NASDAQ', 'type': 'stock'},
            {'symbol': 'NFLX', 'name': 'Netflix Inc.', 'exchange': 'NASDAQ', 'exchangeShortName': 'NASDAQ', 'type': 'stock'},
            {'symbol': 'CRM', 'name': 'Salesforce Inc.', 'exchange': 'NYSE', 'exchangeShortName': 'NYSE', 'type': 'stock'},
            {'symbol': 'XOM', 'name': 'Exxon Mobil Corp.', 'exchange': 'NYSE', 'exchangeShortName': 'NYSE', 'type': 'stock'},
            {'symbol': 'WMT', 'name': 'Walmart Inc.', 'exchange': 'NYSE', 'exchangeShortName': 'NYSE', 'type': 'stock'}
        ]
        
        # Apply limit if specified
        if limit:
            companies = predefined_companies[:limit]
        else:
            companies = predefined_companies
        
        self.logger.info(f"Using {len(companies)} predefined companies")
        
        # Save company list
        companies_df = pd.DataFrame(companies)
        save_data(companies_df, 'company_list.csv', config.RAW_DATA_DIR)
        
        return companies
    
    def get_financial_statements(self, ticker: str, statement_type: str, 
                                period: str = 'annual', limit: int = None) -> Optional[pd.DataFrame]:
        """
        Fetch financial statements for a specific company
        
        Args:
            ticker: Company ticker symbol
            statement_type: Type of statement ('income', 'balance', 'cash_flow')
            period: Period type ('annual' or 'quarterly')
            limit: Maximum number of periods to fetch
            
        Returns:
            DataFrame with financial statement data or None if failed
        """
        endpoint_map = {
            'income': config.ENDPOINTS['income_statement'],
            'balance': config.ENDPOINTS['balance_sheet'],
            'cash_flow': config.ENDPOINTS['cash_flow']
        }
        
        if statement_type not in endpoint_map:
            self.logger.error(f"Invalid statement type: {statement_type}")
            return None
        
        url = f"{self.base_url}{endpoint_map[statement_type]}/{ticker}"
        params = {
            'apikey': self.api_key,
            'period': period,
            'limit': limit or config.MAX_PERIODS
        }
        
        self.logger.debug(f"Fetching {period} {statement_type} statement for {ticker}")
        
        data = safe_api_request(url, params)
        
        if not data:
            self.logger.warning(f"Failed to fetch {statement_type} statement for {ticker}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        if df.empty:
            self.logger.warning(f"No {statement_type} data found for {ticker}")
            return None
        
        # Standardize date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=False)
        
        # Add metadata
        df['ticker'] = ticker
        df['statement_type'] = statement_type
        df['period'] = period
        
        return df
    
    def get_financial_ratios(self, ticker: str, period: str = 'annual', 
                            limit: int = None) -> Optional[pd.DataFrame]:
        """
        Fetch financial ratios for a specific company
        
        Args:
            ticker: Company ticker symbol
            period: Period type ('annual' or 'quarterly')
            limit: Maximum number of periods to fetch
            
        Returns:
            DataFrame with financial ratios or None if failed
        """
        url = f"{self.base_url}{config.ENDPOINTS['ratios']}/{ticker}"
        params = {
            'apikey': self.api_key,
            'period': period,
            'limit': limit or config.MAX_PERIODS
        }
        
        self.logger.debug(f"Fetching {period} financial ratios for {ticker}")
        
        data = safe_api_request(url, params)
        
        if not data:
            self.logger.warning(f"Failed to fetch financial ratios for {ticker}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        if df.empty:
            self.logger.warning(f"No financial ratios found for {ticker}")
            return None
        
        # Standardize date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=False)
        
        # Add metadata
        df['ticker'] = ticker
        df['period'] = period
        
        return df
    
    def collect_company_data(self, ticker: str, save_raw: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Collect all financial data for a specific company
        
        Args:
            ticker: Company ticker symbol
            save_raw: Whether to save raw data to files
            
        Returns:
            Dictionary containing all collected data
        """
        self.logger.info(f"Collecting data for company: {ticker}")
        
        company_data = {}
        
        # Collect financial statements
        for period in config.PERIODS:
            # Income Statement
            income_stmt = self.get_financial_statements(ticker, 'income', period)
            if income_stmt is not None:
                company_data[f'income_{period}'] = income_stmt
                if save_raw:
                    save_data(income_stmt, f'{ticker}_income_{period}.csv', config.RAW_DATA_DIR)
            
            # Balance Sheet
            balance_sheet = self.get_financial_statements(ticker, 'balance', period)
            if balance_sheet is not None:
                company_data[f'balance_{period}'] = balance_sheet
                if save_raw:
                    save_data(balance_sheet, f'{ticker}_balance_{period}.csv', config.RAW_DATA_DIR)
            
            # Cash Flow Statement
            cash_flow = self.get_financial_statements(ticker, 'cash_flow', period)
            if cash_flow is not None:
                company_data[f'cash_flow_{period}'] = cash_flow
                if save_raw:
                    save_data(cash_flow, f'{ticker}_cash_flow_{period}.csv', config.RAW_DATA_DIR)
            
            # Financial Ratios
            ratios = self.get_financial_ratios(ticker, period)
            if ratios is not None:
                company_data[f'ratios_{period}'] = ratios
                if save_raw:
                    save_data(ratios, f'{ticker}_ratios_{period}.csv', config.RAW_DATA_DIR)
        
        self.logger.info(f"Collected {len(company_data)} datasets for {ticker}")
        return company_data
    
    def collect_bulk_data(self, companies: List[Dict], max_companies: int = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Collect data for multiple companies
        
        Args:
            companies: List of company dictionaries
            max_companies: Maximum number of companies to process
            
        Returns:
            Dictionary mapping ticker to company data
        """
        if max_companies:
            companies = companies[:max_companies]
        
        self.logger.info(f"Starting bulk data collection for {len(companies)} companies")
        
        all_company_data = {}
        successful_companies = 0
        
        for i, company in enumerate(companies):
            ticker = company['symbol']
            
            try:
                self.logger.info(f"Processing company {i+1}/{len(companies)}: {ticker}")
                
                company_data = self.collect_company_data(ticker, save_raw=True)
                
                if company_data:
                    all_company_data[ticker] = company_data
                    successful_companies += 1
                    self.logger.info(f"Successfully collected data for {ticker}")
                else:
                    self.logger.warning(f"No data collected for {ticker}")
                
                # Respect API rate limits
                if i < len(companies) - 1:  # Don't sleep after the last request
                    time.sleep(config.REQUEST_DELAY)
                    
            except Exception as e:
                self.logger.error(f"Error collecting data for {ticker}: {e}")
                continue
        
        self.logger.info(f"Bulk data collection completed. Successfully processed {successful_companies}/{len(companies)} companies")
        
        return all_company_data
    
    def create_data_summary(self, all_company_data: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """
        Create a summary of collected data
        
        Args:
            all_company_data: Dictionary containing all company data
            
        Returns:
            DataFrame with data collection summary
        """
        summary_data = []
        
        for ticker, company_data in all_company_data.items():
            for dataset_name, dataset in company_data.items():
                if isinstance(dataset, pd.DataFrame) and not dataset.empty:
                    summary_data.append({
                        'ticker': ticker,
                        'dataset': dataset_name,
                        'rows': len(dataset),
                        'columns': len(dataset.columns),
                        'date_range_start': dataset['date'].min() if 'date' in dataset.columns else None,
                        'date_range_end': dataset['date'].max() if 'date' in dataset.columns else None,
                        'missing_values': dataset.isnull().sum().sum()
                    })
        
        summary_df = pd.DataFrame(summary_data)
        
        if not summary_df.empty:
            save_data(summary_df, 'data_collection_summary.csv', config.RAW_DATA_DIR)
            self.logger.info("Data collection summary saved")
        
        return summary_df

def main():
    """Main function for data collection"""
    try:
        # Initialize collector
        collector = FMPDataCollector()
        
        # Get company list
        companies = collector.get_company_list(limit=config.MAX_COMPANIES)
        
        if not companies:
            print("No companies found. Please check your API key and internet connection.")
            return
        
        # Collect data for all companies
        all_company_data = collector.collect_bulk_data(companies)
        
        # Create summary
        summary = collector.create_data_summary(all_company_data)
        
        print(f"\nData collection completed successfully!")
        print(f"Processed {len(all_company_data)} companies")
        print(f"Total datasets collected: {len(summary)}")
        
        if not summary.empty:
            print(f"Data summary saved to: {config.RAW_DATA_DIR}/data_collection_summary.csv")
        
    except Exception as e:
        print(f"Error during data collection: {e}")
        raise

if __name__ == "__main__":
    main()
