"""
Model Trainer for training machine learning models on historical market data
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.models.catboost_target_model import CatBoostTargetModel
from src.models.target_model import TargetModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains machine learning models on historical market data
    """
    
    def __init__(self, model_type: str = 'catboost', model_params: Dict = None):
        """
        Initialize the model trainer
        
        Parameters:
        - model_type: Type of model to train ('catboost', 'heuristic')
        - model_params: Parameters for model initialization
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'price_change'
        
        logger.info(f"ModelTrainer initialized for {model_type} model")

    def train_model(self, df: pd.DataFrame) -> TargetModel:
        """
        Train the model using the provided DataFrame.

        Parameters:
        df (pd.DataFrame): The historical data to train the model.

        Returns:
        model: The trained model.
        """
        logger.info("Starting model training...")
        
        if df.empty:
            raise ValueError("Training data is empty")
        
        # Prepare training data
        X, y = self._prepare_training_data(df)
        
        if len(X) < 10:
            logger.warning(f"Very small training dataset: {len(X)} samples")
        
        # Create model instance
        self.model = self._create_model()
        
        # Train the model
        self.model.train(X, y)
        
        logger.info(f"Model training completed with {len(X)} samples")
        
        return self.model

    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare features and targets for training
        
        Parameters:
        - df: Raw historical data
        
        Returns:
        - Tuple of (features, targets)
        """
        logger.info("Preparing training data...")
        
        # Sort by symbol and timestamp
        df = df.sort_values(['symbol', 'timestamp'])
        
        # Create target variable (future price change)
        df = self._create_target_variable(df)
        
        # Select feature columns
        self.feature_columns = self._select_features(df)
        
        # Prepare features
        X = df[self.feature_columns].copy()
        
        # Handle missing values
        X = X.ffill().fillna(0)
        
        # Get targets
        y = df[self.target_column].values
        
        # Remove rows with NaN targets
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"Prepared {len(X)} training samples with {len(self.feature_columns)} features")
        
        return X, y
    
    def _create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variable for prediction (future price change percentage)
        
        Parameters:
        - df: Historical data
        
        Returns:
        - DataFrame with target variable added
        """
        # Calculate future price change (next period)
        df = df.copy()
        
        # Group by symbol to avoid mixing data across different trading pairs
        def calculate_price_change(group):
            if 'mid_price' in group.columns:
                # Calculate percentage change to next period
                group['future_price'] = group['mid_price'].shift(-1)
                group[self.target_column] = (
                    (group['future_price'] - group['mid_price']) / group['mid_price'] * 100
                )
            else:
                group[self.target_column] = 0
            return group
        
        df = df.groupby('symbol').apply(calculate_price_change).reset_index(drop=True)
        
        # Remove the last row for each symbol (no future price available)
        df = df.groupby('symbol').apply(lambda x: x.iloc[:-1]).reset_index(drop=True)
        
        return df
    
    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """
        Select relevant features for training
        
        Parameters:
        - df: Historical data
        
        Returns:
        - List of feature column names
        """
        # Define potential features
        potential_features = [
            'bid_price', 'ask_price', 'mid_price', 'bid_size', 'ask_size',
            'spread_absolute', 'spread_percentage', 'spread_mid_price',
            'buy_volume', 'sell_volume', 'total_volume', 'volume_imbalance',
            'bid_depth', 'ask_depth', 'total_depth', 'depth_imbalance',
            'bid_levels', 'ask_levels', 'order_changes',
            'price_change', 'price_change_abs', 'price_volatility', 'price_trend',
            'volume_ma', 'volume_ratio', 'spread_ma', 'spread_volatility'
        ]
        
        # Select features that exist in the data
        available_features = [col for col in potential_features if col in df.columns]
        
        # Remove target-related features to avoid data leakage
        features_to_exclude = [self.target_column, 'future_price', 'timestamp', 'symbol', 'file_date']
        selected_features = [col for col in available_features if col not in features_to_exclude]
        
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        
        return selected_features
    
    def _create_model(self) -> TargetModel:
        """
        Create model instance based on model_type
        
        Returns:
        - Model instance
        """
        if self.model_type == 'catboost':
            # Default CatBoost parameters optimized for financial data
            default_params = {
                'iterations': 500,
                'learning_rate': 0.1,
                'depth': 6,
                'l2_leaf_reg': 3.0,
                'random_seed': 42,
                'verbose': False
            }
            default_params.update(self.model_params)
            
            return CatBoostTargetModel(**default_params)
        
        elif self.model_type == 'heuristic':
            from src.models.heuristic_model import HeuristicModel
            return HeuristicModel(**self.model_params)
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def evaluate_model(self, df: pd.DataFrame) -> Dict:
        """
        Evaluate the model using the provided DataFrame.

        Parameters:
        df (pd.DataFrame): The historical data to evaluate the model.

        Returns:
        dict: A dictionary containing evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info("Evaluating model...")
        
        # Prepare evaluation data
        X, y = self._prepare_training_data(df)
        
        # Evaluate using the model's built-in evaluation
        metrics = self.model.evaluate(X, y)
        
        # Add additional metrics
        metrics.update({
            'evaluation_samples': len(X),
            'evaluation_date': datetime.now().isoformat(),
            'model_type': self.model_type,
            'feature_count': len(self.feature_columns) if self.feature_columns else 0
        })
        
        logger.info(f"Model evaluation completed. RMSE: {metrics.get('rmse', 'N/A'):.4f}")
        
        return metrics

    def log_model(self, model: TargetModel, metrics: Dict):
        """
        Log the trained model and its evaluation metrics.

        Parameters:
        model: The trained model to be logged.
        metrics (dict): A dictionary containing evaluation metrics.

        Returns:
        None
        """
        logger.info("Logging model and metrics...")
        
        # Log basic model info
        if hasattr(model, 'get_model_info'):
            model_info = model.get_model_info()
            logger.info(f"Model Type: {model_info.get('model_type', 'Unknown')}")
            logger.info(f"Is Trained: {model_info.get('is_trained', False)}")
        
        # Log metrics
        logger.info("Training Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Log feature importance if available
        if hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
            if importance:
                logger.info("Top 10 Feature Importances:")
                for i, (feature, score) in enumerate(list(importance.items())[:10]):
                    logger.info(f"  {i+1}. {feature}: {score:.4f}")
    
    def save_model(self, model: TargetModel, filepath: str):
        """
        Save the trained model to file
        
        Parameters:
        - model: Trained model to save
        - filepath: Path to save the model
        """
        try:
            model.save_model(filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath: str) -> TargetModel:
        """
        Load a trained model from file
        
        Parameters:
        - filepath: Path to the saved model
        
        Returns:
        - Loaded model instance
        """
        try:
            # Create model instance
            model = self._create_model()
            model.load_model(filepath)
            
            logger.info(f"Model loaded from {filepath}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_training_summary(self) -> Dict:
        """
        Get summary of training configuration
        
        Returns:
        - Dictionary with training summary
        """
        return {
            'model_type': self.model_type,
            'model_params': self.model_params,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'is_trained': self.model is not None and getattr(self.model, 'is_trained', False)
        }
