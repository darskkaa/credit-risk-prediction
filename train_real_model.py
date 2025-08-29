#!/usr/bin/env python3
"""
Real Machine Learning Model Training for Credit Default Risk
Using actual financial data and proper ML techniques
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealMLModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = []
        
    def load_and_prepare_data(self):
        """Load real financial data and create features"""
        logger.info("Loading real financial data...")
        
        # List of companies we have data for
        companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        
        all_features = []
        all_labels = []
        
        for ticker in companies:
            try:
                # Load financial statements
                income_file = f"data/raw/{ticker}_income_annual.csv"
                balance_file = f"data/raw/{ticker}_balance_annual.csv"
                cashflow_file = f"data/raw/{ticker}_cash_flow_annual.csv"
                
                if all(os.path.exists(f) for f in [income_file, balance_file, cashflow_file]):
                    income_df = pd.read_csv(income_file)
                    balance_df = pd.read_csv(balance_file)
                    cashflow_df = pd.read_csv(cashflow_file)
                    
                    # Create features for each year
                    for i in range(len(income_df)):
                        features = self.engineer_features(
                            income_df.iloc[i],
                            balance_df.iloc[i], 
                            cashflow_df.iloc[i],
                            ticker
                        )
                        
                        if features is not None:
                            all_features.append(features)
                            # Create labels based on known credit quality
                            label = self.get_credit_label(ticker)
                            all_labels.append(label)
                            
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No valid financial data found!")
            
        features_df = pd.DataFrame(all_features)
        labels = np.array(all_labels)
        
        logger.info(f"Created {len(features_df)} samples with {len(features_df.columns)} features")
        self.feature_names = list(features_df.columns)
        
        return features_df, labels
    
    def engineer_features(self, income, balance, cashflow, ticker):
        """Engineer comprehensive financial features"""
        try:
            # Basic financial metrics
            total_assets = balance.get('totalAssets', 0)
            total_liabilities = balance.get('totalLiabilities', 0)
            total_equity = balance.get('totalEquity', 0)
            current_assets = balance.get('currentAssets', 0)
            current_liabilities = balance.get('currentLiabilities', 0)
            
            revenue = income.get('revenue', 0)
            net_income = income.get('netIncome', 0)
            operating_income = income.get('operatingIncome', 0)
            interest_expense = income.get('interestExpense', 0)
            ebitda = income.get('ebitda', 0)
            
            operating_cf = cashflow.get('operatingCashFlow', 0)
            investing_cf = cashflow.get('investingCashFlow', 0)
            financing_cf = cashflow.get('financingCashFlow', 0)
            
            # Skip if missing critical data
            if total_assets <= 0 or revenue <= 0:
                return None
            
            # Calculate comprehensive ratios
            features = {
                # Leverage ratios
                'debt_ratio': total_liabilities / total_assets,
                'debt_to_equity': total_liabilities / max(total_equity, 1),
                'equity_ratio': total_equity / total_assets,
                
                # Liquidity ratios
                'current_ratio': current_assets / max(current_liabilities, 1),
                'working_capital_ratio': (current_assets - current_liabilities) / total_assets,
                
                # Profitability ratios
                'roa': net_income / total_assets,
                'roe': net_income / max(total_equity, 1),
                'profit_margin': net_income / revenue,
                'operating_margin': operating_income / revenue,
                'ebitda_margin': ebitda / revenue if ebitda > 0 else 0,
                
                # Efficiency ratios
                'asset_turnover': revenue / total_assets,
                'asset_quality': operating_income / total_assets,
                
                # Coverage ratios
                'interest_coverage': operating_income / max(interest_expense, 1),
                'cash_coverage': operating_cf / max(interest_expense, 1),
                
                # Cash flow ratios
                'operating_cf_ratio': operating_cf / revenue,
                'free_cash_flow_ratio': (operating_cf + investing_cf) / revenue,
                'cash_flow_coverage': operating_cf / max(current_liabilities, 1),
                
                # Size and scale (log transformed)
                'log_total_assets': np.log(max(total_assets, 1)),
                'log_revenue': np.log(max(revenue, 1)),
                
                # Volatility indicators
                'cf_to_debt': operating_cf / max(total_liabilities, 1),
                'earnings_quality': operating_cf / max(net_income, 1) if net_income > 0 else 0,
            }
            
            # Industry dummy variables
            features['is_tech'] = 1 if ticker in ['AAPL', 'MSFT', 'GOOGL'] else 0
            features['is_ecommerce'] = 1 if ticker in ['AMZN'] else 0
            
            # Replace infinite values with large numbers
            for key, value in features.items():
                if np.isinf(value) or np.isnan(value):
                    features[key] = 999 if value > 0 else -999
                    
            return features
            
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            return None
    
    def get_credit_label(self, ticker):
        """Get credit quality labels based on real credit ratings"""
        # 0 = Low Risk, 1 = High Risk
        credit_ratings = {
            'AAPL': 0,    # A+ rating - low risk
            'MSFT': 0,    # A+ rating - low risk  
            'GOOGL': 0,   # A rating - low risk
            'AMZN': 1,    # BBB+ rating - higher risk
        }
        return credit_ratings.get(ticker, 0)
    
    def train_models(self, X, y):
        """Train multiple ML models"""
        logger.info("Training ML models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models
        models_to_train = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42)
        }
        
        best_model = None
        best_score = 0
        
        for name, model in models_to_train.items():
            logger.info(f"Training {name}...")
            
            # Train model
            if name == 'logistic_regression':
                model.fit(X_train_scaled, y_train)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Evaluate
            try:
                auc_score = roc_auc_score(y_test, y_pred_proba)
                logger.info(f"{name} AUC: {auc_score:.4f}")
                
                # Cross-validation
                if name == 'logistic_regression':
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='roc_auc')
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc')
                    
                logger.info(f"{name} CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
                # Save model
                self.models[name] = {
                    'model': model,
                    'auc': auc_score,
                    'cv_auc': cv_scores.mean(),
                    'uses_scaling': name == 'logistic_regression'
                }
                
                if cv_scores.mean() > best_score:
                    best_score = cv_scores.mean()
                    best_model = name
                    
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
        
        logger.info(f"Best model: {best_model} with CV AUC: {best_score:.4f}")
        return best_model
    
    def save_models(self):
        """Save trained models and scaler"""
        logger.info("Saving models...")
        
        os.makedirs('models', exist_ok=True)
        
        # Save scaler
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        # Save feature names
        joblib.dump(self.feature_names, 'models/feature_names.pkl')
        
        # Save models
        for name, model_info in self.models.items():
            joblib.dump(model_info, f'models/{name}.pkl')
            
        logger.info(f"Saved {len(self.models)} models to models/ directory")

def main():
    """Train real ML models using actual financial data"""
    trainer = RealMLModelTrainer()
    
    try:
        # Load and prepare data
        X, y = trainer.load_and_prepare_data()
        
        # Train models
        best_model = trainer.train_models(X, y)
        
        # Save models
        trainer.save_models()
        
        logger.info("✅ Real ML model training completed successfully!")
        logger.info(f"🎯 Best performing model: {best_model}")
        logger.info("🔧 Models ready for use in website.py")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == '__main__':
    main()
