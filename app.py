"""
Flask API for Credit Default Risk Prediction Model

This module provides a REST API for making predictions using trained models.
"""

import os
import json
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import config
from utils import setup_logging, create_directories, get_company_info, calculate_risk_score

# HTML template for the API documentation
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Credit Default Risk Prediction API</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; }
        .endpoint { background: #ecf0f1; padding: 20px; margin: 20px 0; border-radius: 5px; }
        .method { background: #3498db; color: white; padding: 5px 10px; border-radius: 3px; display: inline-block; margin-right: 10px; }
        .url { background: #2ecc71; color: white; padding: 5px 10px; border-radius: 3px; font-family: monospace; }
        .example { background: #f8f9fa; padding: 15px; border-left: 4px solid #3498db; margin: 10px 0; }
        code { background: #f1f2f6; padding: 2px 5px; border-radius: 3px; font-family: monospace; }
        .note { background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Credit Default Risk Prediction API</h1>
        
        <div class="note">
            <strong>Note:</strong> This API provides credit default risk predictions using machine learning models trained on financial data from the FMP API.
        </div>
        
        <h2>📊 Available Endpoints</h2>
        
        <div class="endpoint">
            <h3><span class="method">GET</span> <span class="url">/</span></h3>
            <p>API documentation and usage instructions.</p>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> <span class="url">/predict</span></h3>
            <p>Make a prediction for a company using its ticker symbol.</p>
            
            <h4>Request Body:</h4>
            <div class="example">
                <code>{"ticker": "AAPL"}</code>
            </div>
            
            <h4>Response:</h4>
            <div class="example">
                <code>{
  "ticker": "AAPL",
  "company_name": "Apple Inc.",
  "prediction": {
    "default_probability": 0.15,
    "risk_score": "Low",
    "confidence": 0.85
  },
  "timestamp": "2024-01-15T10:30:00Z"
}</code>
            </div>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> <span class="url">/predict_batch</span></h3>
            <p>Make predictions for multiple companies at once.</p>
            
            <h4>Request Body:</h4>
            <div class="example">
                <code>{"tickers": ["AAPL", "MSFT", "GOOGL"]}</code>
            </div>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">GET</span> <span class="url">/models</span></h3>
            <p>Get information about available models.</p>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">GET</span> <span class="url">/health</span></h3>
            <p>Check API health status.</p>
        </div>
        
        <h2>🔧 Usage Examples</h2>
        
        <h3>Python Example:</h3>
        <div class="example">
            <code>
import requests

# Single prediction
response = requests.post('http://localhost:5000/predict', 
                       json={'ticker': 'AAPL'})
result = response.json()
print(f"Default probability: {result['prediction']['default_probability']:.2%}")

# Batch prediction
response = requests.post('http://localhost:5000/predict_batch', 
                       json={'tickers': ['AAPL', 'MSFT']})
results = response.json()
for result in results:
    print(f"{result['ticker']}: {result['prediction']['risk_score']}")
            </code>
        </div>
        
        <h3>cURL Example:</h3>
        <div class="example">
            <code>
curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"ticker": "AAPL"}'
            </code>
        </div>
        
        <h2>📈 Model Information</h2>
        <p>The API uses the best performing model from training, which may be:</p>
        <ul>
            <li>Logistic Regression</li>
            <li>Random Forest</li>
            <li>XGBoost</li>
        </ul>
        
        <h2>⚠️ Important Notes</h2>
        <ul>
            <li>Ensure the model has been trained before using the API</li>
            <li>Valid ticker symbols are required (e.g., AAPL, MSFT, GOOGL)</li>
            <li>Predictions are based on historical financial data</li>
            <li>Results should not be used as the sole basis for investment decisions</li>
        </ul>
    </div>
</body>
</html>
"""

class CreditDefaultPredictor:
    """Main prediction class for credit default risk"""
    
    def __init__(self):
        self.logger = setup_logging(__name__)
        self.models = {}
        self.best_model = None
        self.feature_names = []
        self.scaler = None
        
        create_directories()
        self.load_models()
    
    def load_models(self) -> None:
        """Load trained models and preprocessing components"""
        try:
            # Load feature names
            feature_names_path = os.path.join(config.PROCESSED_DATA_DIR, 'feature_names.csv')
            if os.path.exists(feature_names_path):
                feature_names_df = pd.read_csv(feature_names_path)
                self.feature_names = feature_names_df['feature_name'].tolist()
                self.logger.info(f"Loaded {len(self.feature_names)} feature names")
            
            # Load models
            for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
                model_path = os.path.join(config.MODELS_DIR, f'{model_name}_model.pkl')
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    self.models[model_name] = model
                    self.logger.info(f"Loaded {model_name} model")
            
            # Load evaluation results to find best model
            results_path = os.path.join(config.PROCESSED_DATA_DIR, 'model_evaluation_results.csv')
            if os.path.exists(results_path):
                results_df = pd.read_csv(results_path, index_col=0)
                best_model_name = results_df['roc_auc'].idxmax()
                if best_model_name in self.models:
                    self.best_model = self.models[best_model_name]
                    self.logger.info(f"Best model: {best_model_name}")
            
            if not self.best_model:
                # Use first available model
                if self.models:
                    self.best_model = list(self.models.values())[0]
                    self.logger.warning("No best model found, using first available model")
                else:
                    raise ValueError("No models found. Please train models first.")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise
    
    def fetch_company_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch financial data for a company from FMP API
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            DataFrame with company financial data or None if failed
        """
        try:
            # This would integrate with the data collection module
            # For now, return a placeholder structure
            self.logger.info(f"Fetching data for {ticker}")
            
            # In a real implementation, this would call the FMP API
            # and return the actual financial data
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {e}")
            return None
    
    def preprocess_company_data(self, company_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Preprocess company data for prediction
        
        Args:
            company_data: Raw company financial data
            
        Returns:
            Preprocessed features DataFrame or None if failed
        """
        try:
            # This would implement the same preprocessing pipeline
            # used during training
            self.logger.info("Preprocessing company data")
            
            # Placeholder implementation
            return None
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {e}")
            return None
    
    def make_prediction(self, ticker: str) -> Dict[str, Any]:
        """
        Make a prediction for a company
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            Dictionary with prediction results
        """
        try:
            self.logger.info(f"Making prediction for {ticker}")
            
            # Get company information
            company_info = get_company_info(ticker)
            company_name = company_info.get('companyName', ticker)
            
            # For demonstration purposes, generate a mock prediction
            # In production, this would use real data and the trained model
            default_probability = np.random.uniform(0.05, 0.35)
            risk_score = calculate_risk_score(default_probability)
            confidence = np.random.uniform(0.7, 0.95)
            
            prediction_result = {
                'ticker': ticker,
                'company_name': company_name,
                'prediction': {
                    'default_probability': round(default_probability, 4),
                    'risk_score': risk_score,
                    'confidence': round(confidence, 4)
                },
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'model_used': type(self.best_model).__name__,
                'note': 'This is a demonstration prediction. Real predictions require trained models and live financial data.'
            }
            
            self.logger.info(f"Prediction completed for {ticker}")
            return prediction_result
            
        except Exception as e:
            self.logger.error(f"Error making prediction for {ticker}: {e}")
            return {
                'error': f'Prediction failed: {str(e)}',
                'ticker': ticker,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
    
    def make_batch_predictions(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple companies
        
        Args:
            tickers: List of company ticker symbols
            
        Returns:
            List of prediction results
        """
        self.logger.info(f"Making batch predictions for {len(tickers)} companies")
        
        results = []
        for ticker in tickers:
            result = self.make_prediction(ticker)
            results.append(result)
        
        return results

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize predictor
try:
    predictor = CreditDefaultPredictor()
    app.logger.info("Credit Default Predictor initialized successfully")
except Exception as e:
    app.logger.error(f"Failed to initialize predictor: {e}")
    predictor = None

@app.route('/')
def index():
    """API documentation page"""
    return HTML_TEMPLATE

@app.route('/health')
def health_check():
    """Health check endpoint"""
    if predictor and predictor.best_model:
        status = 'healthy'
        model_status = 'loaded'
    else:
        status = 'unhealthy'
        model_status = 'not_loaded'
    
    return jsonify({
        'status': status,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'model_status': model_status,
        'available_models': list(predictor.models.keys()) if predictor else []
    })

@app.route('/models')
def get_models():
    """Get information about available models"""
    if not predictor:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    models_info = {}
    for name, model in predictor.models.items():
        models_info[name] = {
            'type': type(model).__name__,
            'parameters': str(model.get_params()) if hasattr(model, 'get_params') else 'N/A'
        }
    
    return jsonify({
        'available_models': models_info,
        'best_model': type(predictor.best_model).__name__ if predictor.best_model else None,
        'feature_count': len(predictor.feature_names)
    })

@app.route('/predict', methods=['POST'])
def predict_single():
    """Make a single prediction"""
    if not predictor:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    try:
        data = request.get_json()
        if not data or 'ticker' not in data:
            return jsonify({'error': 'Missing ticker parameter'}), 400
        
        ticker = data['ticker'].upper().strip()
        if not ticker:
            return jsonify({'error': 'Invalid ticker symbol'}), 400
        
        result = predictor.make_prediction(ticker)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Error in prediction endpoint: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Make batch predictions"""
    if not predictor:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    try:
        data = request.get_json()
        if not data or 'tickers' not in data:
            return jsonify({'error': 'Missing tickers parameter'}), 400
        
        tickers = data['tickers']
        if not isinstance(tickers, list) or len(tickers) == 0:
            return jsonify({'error': 'Tickers must be a non-empty list'}), 400
        
        # Limit batch size
        if len(tickers) > 50:
            return jsonify({'error': 'Maximum batch size is 50 companies'}), 400
        
        # Clean ticker symbols
        cleaned_tickers = [t.upper().strip() for t in tickers if t and t.strip()]
        if not cleaned_tickers:
            return jsonify({'error': 'No valid ticker symbols provided'}), 400
        
        results = predictor.make_batch_predictions(cleaned_tickers)
        
        return jsonify({
            'predictions': results,
            'total_companies': len(results),
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
        
    except Exception as e:
        app.logger.error(f"Error in batch prediction endpoint: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    if predictor:
        app.logger.info("Starting Credit Default Risk Prediction API...")
        app.logger.info(f"Available models: {list(predictor.models.keys())}")
        app.logger.info(f"Best model: {type(predictor.best_model).__name__ if predictor.best_model else 'None'}")
        
        app.run(
            host=config.FLASK_HOST,
            port=config.FLASK_PORT,
            debug=config.FLASK_DEBUG
        )
    else:
        app.logger.error("Cannot start API: Predictor not initialized")
