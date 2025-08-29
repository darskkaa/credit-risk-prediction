"""
Main Execution Script for Credit Default Risk Prediction Model

This script orchestrates the entire machine learning pipeline:
1. Data Collection from FMP API
2. Data Preprocessing and Feature Engineering
3. Model Training and Hyperparameter Tuning
4. Model Evaluation and Visualization
5. Optional: Start the Flask API

Usage:
    python main.py [--skip-collection] [--skip-preprocessing] [--skip-training] [--skip-evaluation] [--start-api]
"""

import os
import sys
import argparse
import time
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils import setup_logging, create_directories

def print_banner():
    """Print project banner"""
    print("=" * 80)
    print("🎯 Credit Default Risk Prediction Model - Complete ML Pipeline")
    print("=" * 80)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Configuration: {config.MAX_COMPANIES} companies, {config.CV_FOLDS} CV folds")
    print("=" * 80)

def print_step_header(step_name: str, step_number: int, total_steps: int):
    """Print step header with progress indicator"""
    print(f"\n{'='*60}")
    print(f"📋 STEP {step_number}/{total_steps}: {step_name}")
    print(f"{'='*60}")

def run_data_collection() -> bool:
    """Run data collection step"""
    try:
        print_step_header("Data Collection from FMP API", 1, 4)
        print("🔄 Fetching company list and financial data...")
        
        # Import here to avoid circular imports
        from data_collection import FMPDataCollector
        
        collector = FMPDataCollector()
        
        # Get company list
        companies = collector.get_company_list(limit=config.MAX_COMPANIES)
        if not companies:
            print("❌ No companies found. Please check your API key and internet connection.")
            return False
        
        print(f"✅ Retrieved {len(companies)} companies")
        
        # Collect data for companies
        all_company_data = collector.collect_bulk_data(companies)
        
        if not all_company_data:
            print("❌ No company data collected")
            return False
        
        # Create summary
        summary = collector.create_data_summary(all_company_data)
        
        print(f"✅ Data collection completed successfully!")
        print(f"   Processed {len(all_company_data)} companies")
        print(f"   Total datasets: {len(summary)}")
        print(f"   Data saved to: {config.RAW_DATA_DIR}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data collection failed: {e}")
        return False

def run_data_preprocessing() -> bool:
    """Run data preprocessing step"""
    try:
        print_step_header("Data Preprocessing and Feature Engineering", 2, 4)
        print("🔄 Loading and preprocessing financial data...")
        
        # Import here to avoid circular imports
        from data_preprocessing import DataPreprocessor
        
        # Check if raw data exists
        company_list_path = os.path.join(config.RAW_DATA_DIR, 'company_list.csv')
        if not os.path.exists(company_list_path):
            print("❌ Company list not found. Please run data collection first.")
            return False
        
        preprocessor = DataPreprocessor()
        
        # Load company list
        import pandas as pd
        company_list_df = pd.read_csv(company_list_path)
        company_list = company_list_df['symbol'].tolist()
        
        # Limit companies for processing
        company_list = company_list[:min(len(company_list), 50)]
        print(f"📊 Processing {len(company_list)} companies...")
        
        # Preprocess all data
        features, labels, feature_names = preprocessor.preprocess_all_data(company_list)
        
        if features.empty:
            print("❌ Data preprocessing failed")
            return False
        
        print(f"✅ Data preprocessing completed successfully!")
        print(f"   Final dataset shape: {features.shape}")
        print(f"   Number of features: {len(feature_names)}")
        print(f"   Class distribution: {labels.value_counts().to_dict()}")
        print(f"   Preprocessed data saved to: {config.PROCESSED_DATA_DIR}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        return False

def run_model_training() -> bool:
    """Run model training step"""
    try:
        print_step_header("Model Training and Hyperparameter Tuning", 3, 4)
        print("🔄 Training machine learning models...")
        
        # Import here to avoid circular imports
        from model_training import ModelTrainer
        
        # Check if preprocessed data exists
        features_path = os.path.join(config.PROCESSED_DATA_DIR, 'preprocessed_features.csv')
        if not os.path.exists(features_path):
            print("❌ Preprocessed data not found. Please run data preprocessing first.")
            return False
        
        trainer = ModelTrainer()
        
        # Run training pipeline
        trained_models, evaluation_results = trainer.train_pipeline()
        
        if not trained_models:
            print("❌ Model training failed")
            return False
        
        print(f"✅ Model training completed successfully!")
        print(f"   Trained {len(trained_models)} models")
        
        # Display results
        print("\n📊 Model Performance Summary:")
        print("-" * 50)
        
        for model_name, metrics in evaluation_results.items():
            print(f"\n{model_name.upper()}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # Show best model
        best_model = trainer.select_best_model(evaluation_results)
        print(f"\n🏆 Best Model: {best_model}")
        print(f"📁 Models saved to: {config.MODELS_DIR}")
        print(f"📊 Results saved to: {config.PROCESSED_DATA_DIR}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False

def run_model_evaluation() -> bool:
    """Run model evaluation step"""
    try:
        print_step_header("Model Evaluation and Visualization", 4, 4)
        print("🔄 Evaluating models and generating visualizations...")
        
        # Import here to avoid circular imports
        from model_evaluation import ModelEvaluator
        
        # Check if trained models exist
        models_dir = config.MODELS_DIR
        if not os.path.exists(models_dir) or not os.listdir(models_dir):
            print("❌ Trained models not found. Please run model training first.")
            return False
        
        evaluator = ModelEvaluator()
        
        # Run complete evaluation
        all_reports = evaluator.run_complete_evaluation()
        
        if not all_reports:
            print("❌ Model evaluation failed")
            return False
        
        print(f"✅ Model evaluation completed successfully!")
        print(f"   Evaluated {len(all_reports)} models")
        
        # Display summary
        print("\n📊 Model Performance Summary:")
        print("-" * 60)
        
        for model_name, report in all_reports.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy:      {report['accuracy']:.4f}")
            print(f"  Precision:     {report['precision']:.4f}")
            print(f"  Recall:        {report['recall']:.4f}")
            print(f"  F1-Score:      {report['f1_score']:.4f}")
            print(f"  ROC-AUC:       {report['roc_auc']:.4f}")
            print(f"  Specificity:   {report['specificity']:.4f}")
            print(f"  Sensitivity:   {report['sensitivity']:.4f}")
        
        plots_dir = os.path.join(config.PROCESSED_DATA_DIR, 'plots')
        print(f"\n📈 Plots saved to: {plots_dir}")
        print(f"📊 Detailed report saved to: {config.PROCESSED_DATA_DIR}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model evaluation failed: {e}")
        return False

def start_flask_api():
    """Start the Flask API"""
    try:
        print("\n🚀 Starting Flask API...")
        print("📡 API will be available at: http://localhost:5000")
        print("📚 API documentation: http://localhost:5000/")
        print("⏹️  Press Ctrl+C to stop the API")
        
        # Import and start Flask app
        from app import app
        app.run(
            host=config.FLASK_HOST,
            port=config.FLASK_PORT,
            debug=config.FLASK_DEBUG
        )
        
    except KeyboardInterrupt:
        print("\n⏹️  API stopped by user")
    except Exception as e:
        print(f"❌ Failed to start API: {e}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Credit Default Risk Prediction Pipeline')
    parser.add_argument('--skip-collection', action='store_true', help='Skip data collection step')
    parser.add_argument('--skip-preprocessing', action='store_true', help='Skip data preprocessing step')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training step')
    parser.add_argument('--skip-evaluation', action='store_true', help='Skip model evaluation step')
    parser.add_argument('--start-api', action='store_true', help='Start Flask API after pipeline completion')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Create necessary directories
    create_directories()
    
    # Initialize logging
    logger = setup_logging(__name__)
    logger.info("Starting Credit Default Risk Prediction Pipeline")
    
    # Track execution status
    execution_status = {}
    total_steps = 4
    
    try:
        # Step 1: Data Collection
        if not args.skip_collection:
            execution_status['collection'] = run_data_collection()
            if not execution_status['collection']:
                print("\n❌ Pipeline failed at data collection step")
                return False
        else:
            print("\n⏭️  Skipping data collection step")
            execution_status['collection'] = True
        
        # Step 2: Data Preprocessing
        if not args.skip_preprocessing:
            execution_status['preprocessing'] = run_data_preprocessing()
            if not execution_status['preprocessing']:
                print("\n❌ Pipeline failed at data preprocessing step")
                return False
        else:
            print("\n⏭️  Skipping data preprocessing step")
            execution_status['preprocessing'] = True
        
        # Step 3: Model Training
        if not args.skip_training:
            execution_status['training'] = run_model_training()
            if not execution_status['training']:
                print("\n❌ Pipeline failed at model training step")
                return False
        else:
            print("\n⏭️  Skipping model training step")
            execution_status['training'] = True
        
        # Step 4: Model Evaluation
        if not args.skip_evaluation:
            execution_status['evaluation'] = run_model_evaluation()
            if not execution_status['evaluation']:
                print("\n❌ Pipeline failed at model evaluation step")
                return False
        else:
            print("\n⏭️  Skipping model evaluation step")
            execution_status['evaluation'] = True
        
        # Pipeline completed successfully
        print("\n" + "="*80)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Summary of completed steps
        completed_steps = sum(execution_status.values())
        print(f"✅ Completed {completed_steps}/{total_steps} steps:")
        
        step_names = ['Data Collection', 'Data Preprocessing', 'Model Training', 'Model Evaluation']
        for i, (step, status) in enumerate(execution_status.items()):
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {step_names[i]}")
        
        print(f"\n📁 Data saved to: {config.DATA_DIR}")
        print(f"🤖 Models saved to: {config.MODELS_DIR}")
        print(f"📊 Results saved to: {config.PROCESSED_DATA_DIR}")
        
        # Start API if requested
        if args.start_api:
            start_flask_api()
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Pipeline interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with unexpected error: {e}")
        logger.error(f"Pipeline failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
