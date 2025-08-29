"""
Model Evaluation Module for Credit Default Risk Prediction Model

This module provides comprehensive model evaluation, visualization,
and detailed performance reporting.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score
)
from sklearn.model_selection import learning_curve
import config
from utils import setup_logging, create_directories, load_data

class ModelEvaluator:
    """Handles comprehensive model evaluation and visualization"""
    
    def __init__(self):
        self.logger = setup_logging(__name__)
        self.models = {}
        self.evaluation_results = {}
        self.feature_names = []
        create_directories()
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_models_and_data(self) -> Tuple[Dict[str, Any], pd.DataFrame, pd.Series]:
        """
        Load trained models and test data
        
        Returns:
            Tuple of (models, features, labels)
        """
        self.logger.info("Loading models and data...")
        
        # Load models
        for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
            model_path = os.path.join(config.MODELS_DIR, f'{model_name}_model.pkl')
            if os.path.exists(model_path):
                try:
                    model = joblib.load(model_path)
                    self.models[model_name] = model
                    self.logger.info(f"Loaded {model_name} model")
                except Exception as e:
                    self.logger.warning(f"Failed to load {model_name} model: {e}")
        
        if not self.models:
            raise ValueError("No models found. Please run model_training.py first.")
        
        # Load preprocessed data
        features = load_data('preprocessed_features.csv')
        labels = load_data('preprocessed_labels.csv')['default']
        
        # Load feature names
        feature_names_df = load_data('feature_names.csv')
        self.feature_names = feature_names_df['feature_name'].tolist()
        
        # Split data (same split as training)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=labels
        )
        
        self.logger.info(f"Loaded {len(self.models)} models and test data: {X_test.shape}")
        return self.models, X_test, y_test
    
    def generate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 model_name: str, save_path: str = None) -> None:
        """
        Generate and display confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save_path: Path to save the plot
        """
        self.logger.info(f"Generating confusion matrix for {model_name}")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Default', 'Default'],
                   yticklabels=['Non-Default', 'Default'])
        plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def generate_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                          model_name: str, save_path: str = None) -> None:
        """
        Generate and display ROC curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save_path: Path to save the plot
        """
        self.logger.info(f"Generating ROC curve for {model_name}")
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name.replace("_", " ").title()}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def generate_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                      model_name: str, save_path: str = None) -> None:
        """
        Generate and display precision-recall curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save_path: Path to save the plot
        """
        self.logger.info(f"Generating precision-recall curve for {model_name}")
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, 
                label=f'{model_name} (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name.replace("_", " ").title()}')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Precision-recall curve saved to {save_path}")
        
        plt.show()
    
    def generate_feature_importance_plot(self, model: Any, model_name: str, 
                                       save_path: str = None) -> None:
        """
        Generate feature importance plot for tree-based models
        
        Args:
            model: Trained model
            model_name: Name of the model
            save_path: Path to save the plot
        """
        self.logger.info(f"Generating feature importance plot for {model_name}")
        
        # Check if model has feature_importances_ attribute
        if not hasattr(model, 'feature_importances_'):
            self.logger.warning(f"{model_name} does not support feature importance")
            return
        
        # Get feature importance
        importance = model.feature_importances_
        
        if len(importance) != len(self.feature_names):
            self.logger.warning(f"Feature importance length mismatch for {model_name}")
            return
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        # Plot top 20 features
        top_features = importance_df.tail(20)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 20 Feature Importance - {model_name.replace("_", " ").title()}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def generate_model_comparison_plot(self, save_path: str = None) -> None:
        """
        Generate comparison plot for all models
        
        Args:
            save_path: Path to save the plot
        """
        self.logger.info("Generating model comparison plot")
        
        # Load evaluation results
        results_path = os.path.join(config.PROCESSED_DATA_DIR, 'model_evaluation_results.csv')
        if not os.path.exists(results_path):
            self.logger.warning("Evaluation results not found. Cannot generate comparison plot.")
            return
        
        results_df = pd.read_csv(results_path, index_col=0)
        
        # Create comparison plot
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                ax = axes[i]
                results_df[metric].plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title(f'{metric.upper()} Comparison')
                ax.set_ylabel(metric.upper())
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for j, v in enumerate(results_df[metric]):
                    ax.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Remove extra subplot
        if len(metrics) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.suptitle('Model Performance Comparison', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def generate_detailed_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                y_pred_proba: np.ndarray, model_name: str) -> Dict[str, Any]:
        """
        Generate detailed performance report for a model
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            
        Returns:
            Dictionary with detailed metrics
        """
        self.logger.info(f"Generating detailed report for {model_name}")
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        report = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'average_precision': average_precision_score(y_true, y_pred_proba)
        }
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        report['confusion_matrix'] = {
            'true_negatives': cm[0, 0],
            'false_positives': cm[0, 1],
            'false_negatives': cm[1, 0],
            'true_positives': cm[1, 1]
        }
        
        # Calculate additional metrics
        tn, fp, fn, tp = cm.ravel()
        report['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        report['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Classification report
        report['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
        
        return report
    
    def save_evaluation_report(self, all_reports: Dict[str, Dict[str, Any]]) -> str:
        """
        Save comprehensive evaluation report
        
        Args:
            all_reports: Dictionary of detailed reports for all models
            
        Returns:
            Path to saved report
        """
        self.logger.info("Saving evaluation report...")
        
        # Create summary DataFrame
        summary_data = []
        for model_name, report in all_reports.items():
            summary_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{report['accuracy']:.4f}",
                'Precision': f"{report['precision']:.4f}",
                'Recall': f"{report['recall']:.4f}",
                'F1-Score': f"{report['f1_score']:.4f}",
                'ROC-AUC': f"{report['roc_auc']:.4f}",
                'Average Precision': f"{report['average_precision']:.4f}",
                'Specificity': f"{report['specificity']:.4f}",
                'Sensitivity': f"{report['sensitivity']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to file
        report_path = os.path.join(config.PROCESSED_DATA_DIR, 'detailed_evaluation_report.csv')
        summary_df.to_csv(report_path, index=False)
        
        self.logger.info(f"Evaluation report saved to {report_path}")
        return report_path
    
    def run_complete_evaluation(self) -> Dict[str, Dict[str, Any]]:
        """
        Run complete evaluation pipeline
        
        Returns:
            Dictionary of detailed reports for all models
        """
        self.logger.info("Starting complete model evaluation...")
        
        try:
            # Load models and data
            models, X_test, y_test = self.load_models_and_data()
            
            # Create plots directory
            plots_dir = os.path.join(config.PROCESSED_DATA_DIR, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            all_reports = {}
            
            # Evaluate each model
            for model_name, model in models.items():
                self.logger.info(f"Evaluating {model_name}...")
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Generate plots
                self.generate_confusion_matrix(
                    y_test, y_pred, model_name,
                    os.path.join(plots_dir, f'{model_name}_confusion_matrix.png')
                )
                
                self.generate_roc_curve(
                    y_test, y_pred_proba, model_name,
                    os.path.join(plots_dir, f'{model_name}_roc_curve.png')
                )
                
                self.generate_precision_recall_curve(
                    y_test, y_pred_proba, model_name,
                    os.path.join(plots_dir, f'{model_name}_precision_recall.png')
                )
                
                # Generate feature importance plot for tree-based models
                if hasattr(model, 'feature_importances_'):
                    self.generate_feature_importance_plot(
                        model, model_name,
                        os.path.join(plots_dir, f'{model_name}_feature_importance.png')
                    )
                
                # Generate detailed report
                report = self.generate_detailed_report(y_test, y_pred, y_pred_proba, model_name)
                all_reports[model_name] = report
            
            # Generate model comparison plot
            self.generate_model_comparison_plot(
                os.path.join(plots_dir, 'model_comparison.png')
            )
            
            # Save evaluation report
            self.save_evaluation_report(all_reports)
            
            self.logger.info("Complete evaluation finished successfully")
            return all_reports
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise

def main():
    """Main function for model evaluation"""
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator()
        
        # Run complete evaluation
        all_reports = evaluator.run_complete_evaluation()
        
        print(f"\nModel evaluation completed successfully!")
        print(f"Evaluated {len(all_reports)} models")
        
        # Display summary
        print("\nModel Performance Summary:")
        print("=" * 60)
        
        for model_name, report in all_reports.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy:      {report['accuracy']:.4f}")
            print(f"  Precision:     {report['precision']:.4f}")
            print(f"  Recall:        {report['recall']:.4f}")
            print(f"  F1-Score:      {report['f1_score']:.4f}")
            print(f"  ROC-AUC:       {report['roc_auc']:.4f}")
            print(f"  Specificity:   {report['specificity']:.4f}")
            print(f"  Sensitivity:   {report['sensitivity']:.4f}")
        
        print(f"\nPlots saved to: {os.path.join(config.PROCESSED_DATA_DIR, 'plots')}")
        print(f"Detailed report saved to: {config.PROCESSED_DATA_DIR}")
        
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        raise

if __name__ == "__main__":
    main()
