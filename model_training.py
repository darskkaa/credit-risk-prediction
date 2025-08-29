"""
Model Training Module for Credit Default Risk Prediction Model

This module handles model training, hyperparameter tuning, and cross-validation
for multiple machine learning algorithms.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import config
from utils import setup_logging, create_directories, save_data, load_data

class ModelTrainer:
    """Handles model training and hyperparameter tuning"""
    
    def __init__(self):
        self.logger = setup_logging(__name__)
        self.models = {}
        self.best_models = {}
        self.feature_names = []
        create_directories()
        
    def load_preprocessed_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load preprocessed features and labels
        
        Returns:
            Tuple of (features, labels)
        """
        self.logger.info("Loading preprocessed data...")
        
        try:
            features = load_data('preprocessed_features.csv')
            labels = load_data('preprocessed_labels.csv')['default']
            
            self.logger.info(f"Loaded data: {features.shape[0]} samples, {features.shape[1]} features")
            return features, labels
            
        except Exception as e:
            self.logger.error(f"Failed to load preprocessed data: {e}")
            raise
    
    def split_data(self, features: pd.DataFrame, labels: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets
        
        Args:
            features: Feature matrix
            labels: Target labels
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.logger.info("Splitting data into train/test sets...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=labels
        )
        
        self.logger.info(f"Training set: {X_train.shape[0]} samples")
        self.logger.info(f"Testing set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize all models to be trained
        
        Returns:
            Dictionary of model instances
        """
        self.logger.info("Initializing models...")
        
        models = {
            'logistic_regression': LogisticRegression(
                random_state=config.RANDOM_STATE,
                max_iter=1000,
                solver='liblinear'
            ),
            'random_forest': RandomForestClassifier(
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            )
        }
        
        # Try to import XGBoost if available
        try:
            import xgboost as xgb
            models['xgboost'] = xgb.XGBClassifier(
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                eval_metric='logloss'
            )
            self.logger.info("XGBoost model initialized")
        except ImportError:
            self.logger.warning("XGBoost not available. Skipping XGBoost model.")
        
        self.models = models
        self.logger.info(f"Initialized {len(models)} models")
        
        return models
    
    def perform_hyperparameter_tuning(self, model_name: str, model: Any, 
                                    X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """
        Perform hyperparameter tuning for a specific model
        
        Args:
            model_name: Name of the model
            model: Model instance
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Best model after hyperparameter tuning
        """
        self.logger.info(f"Performing hyperparameter tuning for {model_name}...")
        
        if model_name not in config.MODEL_PARAMS:
            self.logger.warning(f"No hyperparameters defined for {model_name}. Using default model.")
            model.fit(X_train, y_train)
            return model
        
        param_grid = config.MODEL_PARAMS[model_name]
        
        try:
            # Use GridSearchCV for hyperparameter tuning
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=config.CV_FOLDS,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            self.logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            self.logger.info(f"Best CV score for {model_name}: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            self.logger.warning(f"Hyperparameter tuning failed for {model_name}: {e}. Using default model.")
            model.fit(X_train, y_train)
            return model
    
    def train_model(self, model_name: str, model: Any, 
                   X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """
        Train a single model
        
        Args:
            model_name: Name of the model
            model: Model instance
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        self.logger.info(f"Training {model_name}...")
        
        try:
            # Perform hyperparameter tuning
            best_model = self.perform_hyperparameter_tuning(model_name, model, X_train, y_train)
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                best_model, X_train, y_train,
                cv=config.CV_FOLDS,
                scoring='roc_auc',
                n_jobs=-1
            )
            
            self.logger.info(f"{model_name} CV scores: {cv_scores}")
            self.logger.info(f"{model_name} CV mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            return best_model
            
        except Exception as e:
            self.logger.error(f"Error training {model_name}: {e}")
            raise
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train all models
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of trained models
        """
        self.logger.info("Training all models...")
        
        trained_models = {}
        
        for model_name, model in self.models.items():
            try:
                trained_model = self.train_model(model_name, model, X_train, y_train)
                trained_models[model_name] = trained_model
                self.logger.info(f"Successfully trained {model_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        self.logger.info(f"Successfully trained {len(trained_models)} models")
        return trained_models
    
    def evaluate_model(self, model: Any, model_name: str, 
                      X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a single model
        
        Args:
            model: Trained model
            model_name: Name of the model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Log metrics
        for metric, value in metrics.items():
            self.logger.info(f"{model_name} {metric}: {value:.4f}")
        
        return metrics
    
    def evaluate_all_models(self, trained_models: Dict[str, Any], 
                           X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models
        
        Args:
            trained_models: Dictionary of trained models
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation results for each model
        """
        self.logger.info("Evaluating all models...")
        
        evaluation_results = {}
        
        for model_name, model in trained_models.items():
            try:
                metrics = self.evaluate_model(model, model_name, X_test, y_test)
                evaluation_results[model_name] = metrics
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {e}")
                continue
        
        return evaluation_results
    
    def select_best_model(self, evaluation_results: Dict[str, Dict[str, float]]) -> str:
        """
        Select the best model based on ROC-AUC score
        
        Args:
            evaluation_results: Dictionary of evaluation results
            
        Returns:
            Name of the best model
        """
        self.logger.info("Selecting best model...")
        
        if not evaluation_results:
            raise ValueError("No evaluation results available")
        
        # Find model with highest ROC-AUC
        best_model = max(evaluation_results.items(), key=lambda x: x[1]['roc_auc'])
        best_model_name = best_model[0]
        best_score = best_model[1]['roc_auc']
        
        self.logger.info(f"Best model: {best_model_name} with ROC-AUC: {best_score:.4f}")
        
        return best_model_name
    
    def save_models(self, trained_models: Dict[str, Any], 
                   evaluation_results: Dict[str, Dict[str, float]]) -> None:
        """
        Save trained models and evaluation results
        
        Args:
            trained_models: Dictionary of trained models
            evaluation_results: Dictionary of evaluation results
        """
        self.logger.info("Saving models and results...")
        
        # Save each model
        for model_name, model in trained_models.items():
            model_path = os.path.join(config.MODELS_DIR, f'{model_name}_model.pkl')
            joblib.dump(model, model_path)
            self.logger.info(f"Saved {model_name} to {model_path}")
        
        # Save evaluation results
        results_df = pd.DataFrame(evaluation_results).T
        save_data(results_df, 'model_evaluation_results.csv')
        
        # Save feature names
        feature_names_df = pd.DataFrame({'feature_name': self.feature_names})
        save_data(feature_names_df, 'feature_names.csv')
        
        self.logger.info("Models and results saved successfully")
    
    def train_pipeline(self) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
        """
        Complete training pipeline
        
        Returns:
            Tuple of (trained models, evaluation results)
        """
        self.logger.info("Starting model training pipeline...")
        
        try:
            # Load data
            features, labels = self.load_preprocessed_data()
            
            # Store feature names
            self.feature_names = features.columns.tolist()
            
            # Split data
            X_train, X_test, y_train, y_test = self.split_data(features, labels)
            
            # Initialize models
            self.initialize_models()
            
            # Train all models
            trained_models = self.train_all_models(X_train, y_train)
            
            if not trained_models:
                raise ValueError("No models were successfully trained")
            
            # Evaluate all models
            evaluation_results = self.evaluate_all_models(trained_models, X_test, y_test)
            
            # Select best model
            best_model_name = self.select_best_model(evaluation_results)
            self.best_models[best_model_name] = trained_models[best_model_name]
            
            # Save models and results
            self.save_models(trained_models, evaluation_results)
            
            self.logger.info("Training pipeline completed successfully")
            
            return trained_models, evaluation_results
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            raise

def main():
    """Main function for model training"""
    try:
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Run training pipeline
        trained_models, evaluation_results = trainer.train_pipeline()
        
        print(f"\nModel training completed successfully!")
        print(f"Trained {len(trained_models)} models")
        
        # Display results
        print("\nModel Performance Summary:")
        print("=" * 50)
        
        for model_name, metrics in evaluation_results.items():
            print(f"\n{model_name.upper()}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # Show best model
        best_model = trainer.select_best_model(evaluation_results)
        print(f"\nBest Model: {best_model}")
        print(f"Models saved to: {config.MODELS_DIR}")
        print(f"Results saved to: {config.PROCESSED_DATA_DIR}")
        
    except Exception as e:
        print(f"Error during model training: {e}")
        raise

if __name__ == "__main__":
    main()
