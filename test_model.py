#!/usr/bin/env python3
"""
Test the trained ML models to see what's wrong
"""

import joblib
import pandas as pd
import numpy as np

def test_models():
    print("🔍 Testing ML Models...")
    
    # Load models
    try:
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        lr_model_data = joblib.load('models/logistic_regression.pkl')
        lr_model = lr_model_data['model']
        
        print(f"✅ Models loaded successfully")
        print(f"📊 Feature names: {len(feature_names)} features")
        print(f"🎯 Model type: {type(lr_model)}")
        
        # Test with dummy data similar to JPM
        test_features = {
            'debt_ratio': 0.92,  # High for banks (normal)
            'debt_to_equity': 8.0,  # High for banks (normal)
            'equity_ratio': 0.08,
            'current_ratio': 1.1,  # Typical for banks
            'working_capital_ratio': 0.05,
            'roa': 0.015,  # 1.5% ROA (good for banks)
            'roe': 0.15,  # 15% ROE (excellent for banks)
            'profit_margin': 0.25,  # 25% (excellent)
            'operating_margin': 0.35,
            'ebitda_margin': 0.0,
            'asset_turnover': 0.1,  # Low for banks (normal)
            'asset_quality': 0.015,
            'interest_coverage': 5.0,  # Good coverage
            'cash_coverage': 8.0,
            'operating_cf_ratio': 0.2,
            'free_cash_flow_ratio': 0.1,
            'cash_flow_coverage': 0.5,
            'log_total_assets': np.log(3.7e12),  # JPM size
            'log_revenue': np.log(158e9),
            'cf_to_debt': 0.05,
            'earnings_quality': 1.2,
            'is_tech': 0,  # Not tech
            'is_ecommerce': 0  # Not ecommerce
        }
        
        # Create dataframe
        test_df = pd.DataFrame([test_features])
        print(f"📈 Test features shape: {test_df.shape}")
        
        # Scale features
        test_scaled = scaler.transform(test_df)
        
        # Make prediction
        prediction_proba = lr_model.predict_proba(test_scaled)[0]
        risk_score = prediction_proba[1]  # Probability of class 1 (high risk)
        
        print(f"🎯 RESULT:")
        print(f"   Risk Score: {risk_score:.3f} ({risk_score*100:.1f}%)")
        print(f"   Class 0 (Low Risk): {prediction_proba[0]:.3f}")
        print(f"   Class 1 (High Risk): {prediction_proba[1]:.3f}")
        
        if risk_score > 0.5:
            print(f"❌ PROBLEM: Model predicts JPM-like bank as HIGH RISK!")
            print(f"   This is clearly wrong - JPM has A- credit rating")
        else:
            print(f"✅ Model correctly predicts LOW RISK for JPM-like bank")
            
        # Test with clearly low-risk company (Apple-like)
        print(f"\n🍎 Testing with Apple-like company:")
        apple_features = test_features.copy()
        apple_features.update({
            'debt_ratio': 0.84,  # Apple's actual debt ratio
            'debt_to_equity': 5.4,  # Apple's D/E
            'roa': 0.27,  # 27% ROA (excellent)
            'roe': 1.5,   # 150% ROE (excellent)
            'is_tech': 1,  # Is tech
            'asset_turnover': 1.08  # Higher for tech
        })
        
        apple_df = pd.DataFrame([apple_features])
        apple_scaled = scaler.transform(apple_df)
        apple_pred = lr_model.predict_proba(apple_scaled)[0]
        apple_risk = apple_pred[1]
        
        print(f"   Apple Risk Score: {apple_risk:.3f} ({apple_risk*100:.1f}%)")
        
        if apple_risk > 0.15:
            print(f"❌ PROBLEM: Model predicts Apple as too risky!")
        else:
            print(f"✅ Model correctly predicts Apple as low risk")
            
    except Exception as e:
        print(f"❌ Error testing models: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_models()
