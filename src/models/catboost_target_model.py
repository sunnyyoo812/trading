"""
CatBoost Regressor implementation of TargetModel for trading signal generation
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .target_model import TargetModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CatBoostTargetModel(TargetModel):
    """
    CatBoost regressor implementation for predicting trading signals
    
    This model predicts continuous values that can be interpreted as:
    - Signal strength/confidence
    - Price movement direction and magnitude
    - Position sizing recommendations
    """
    
    def __init__(self, 
                 iterations: int = 1000,
                 learning_rate: float = 0.1,
                 depth: int = 6,
                 l2_leaf_reg: float = 3.0,
                 random_seed: int = 42,
                 verbose: bool = False,
                 **kwargs):
        """
        Initialize CatBoost regressor with trading-optimized parameters
        
        Parameters:
        - iterations: Number of boosting iterations
        - learning_rate: Learning rate for gradient boosting
        - depth: Depth of trees
        - l2_leaf_reg: L2 regularization coefficient
        - random_seed: Random seed for reproducibility
        - verbose: Whether to show training progress
        - **kwargs: Additional CatBoost parameters
        """
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        self.verbose = verbose
        self.additional_params = kwargs
        
        # Initialize model
        self.model = None
        self.feature_names = None
        self.is_trained = False
        self.training_metrics = {}

        logger.info("CatBoostTargetModel initialized")
    
    def _create_model(self) -> CatBoostRegressor:
        """Create CatBoost regressor with specified parameters"""
        params = {
            'iterations': self.iterations,
            'learning_rate': self.learning_rate,
            'depth': self.depth,
            'l2_leaf_reg': self.l2_leaf_reg,
            'random_seed': self.random_seed,
            'verbose': self.verbose,
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'bootstrap_type': 'Bayesian',
            'bagging_temperature': 1.0,
            'od_type': 'Iter',
            'od_wait': 50,
            **self.additional_params
        }
        
        return CatBoostRegressor(**params)
    
    def _prepare_features(self, data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Prepare features for training/prediction
        
        Parameters:
        - data: Input data (dict or DataFrame)
        
        Returns:
        - DataFrame with prepared features
        """
        if isinstance(data, dict):
            # Convert single prediction dict to DataFrame
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Data must be dict or DataFrame")
        
        # Define expected features for trading signals
        expected_features = [
            'bid_price', 'ask_price', 'mid_price', 'bid_size', 'ask_size',
            'spread_absolute', 'spread_percentage', 'spread_mid_price',
            'buy_volume', 'sell_volume', 'total_volume', 'volume_imbalance',
            'buy_orders', 'sell_orders', 'order_imbalance',
            'bid_depth', 'ask_depth', 'total_depth', 'depth_imbalance',
            'weighted_mid_price', 'bid_levels', 'ask_levels'
        ]
        
        # Add missing features with default values
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0.0
        
        # Create additional engineered features
        df = self._engineer_features(df)
        
        # Select and order features consistently
        feature_columns = expected_features + [
            'price_volatility', 'volume_ratio', 'spread_ratio',
            'liquidity_score', 'momentum_score'
        ]
        
        # Ensure all features exist
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        
        # Store feature names for consistency
        if self.feature_names is None:
            self.feature_names = feature_columns
        
        return df[self.feature_names]
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features for better model performance
        
        Parameters:
        - df: Input DataFrame
        
        Returns:
        - DataFrame with additional features
        """
        # Price volatility (using spread as proxy)
        df['price_volatility'] = df['spread_percentage'] / (df['mid_price'] + 1e-8)
        
        # Volume ratio (buy vs sell)
        df['volume_ratio'] = (df['buy_volume'] + 1e-8) / (df['sell_volume'] + 1e-8)
        
        # Spread ratio (absolute spread to mid price)
        df['spread_ratio'] = df['spread_absolute'] / (df['mid_price'] + 1e-8)
        
        # Liquidity score (total depth relative to volume)
        df['liquidity_score'] = df['total_depth'] / (df['total_volume'] + 1e-8)
        
        # Momentum score (combination of volume and depth imbalances)
        df['momentum_score'] = (df['volume_imbalance'] + df['depth_imbalance']) / 2
        
        return df
    
    def train(self, data: Union[Dict, pd.DataFrame], target: Union[List, np.ndarray, pd.Series]):
        """
        Train the CatBoost model
        
        Parameters:
        - data: Training features (dict, DataFrame, or list of dicts)
        - target: Target values for regression
        """
        logger.info("Starting CatBoost model training...")
        
        # Prepare features
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data if isinstance(data, pd.DataFrame) else pd.DataFrame([data])
        
        X = self._prepare_features(df)
        y = np.array(target) if not isinstance(target, np.ndarray) else target
        
        # Validate data
        if len(X) != len(y):
            raise ValueError(f"Feature and target lengths don't match: {len(X)} vs {len(y)}")
        
        if len(X) < 10:
            logger.warning("Very small training dataset. Consider collecting more data.")
        
        # Split data for validation
        if len(X) > 20:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=self.random_seed
            )
            eval_set = (X_val, y_val)
        else:
            X_train, y_train = X, y
            eval_set = None
        
        # Create and train model
        self.model = self._create_model()
        
        try:
            if eval_set is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    use_best_model=True,
                    plot=False
                )
            else:
                self.model.fit(X_train, y_train)
            
            self.is_trained = True
            
            # Calculate training metrics
            y_pred_train = self.model.predict(X_train)
            self.training_metrics = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'train_r2': r2_score(y_train, y_pred_train),
                'n_features': len(self.feature_names),
                'n_samples': len(X_train)
            }
            
            if eval_set is not None:
                y_pred_val = self.model.predict(X_val)
                self.training_metrics.update({
                    'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)),
                    'val_mae': mean_absolute_error(y_val, y_pred_val),
                    'val_r2': r2_score(y_val, y_pred_val)
                })
            
            logger.info(f"Model training completed. Train RMSE: {self.training_metrics['train_rmse']:.4f}")
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise
    
    def predict(self, data: Union[Dict, pd.DataFrame]) -> List[Dict]:
        """
        Generate trading signals based on model predictions
        
        Parameters:
        - data: Market features for prediction
        
        Returns:
        - List of trading signal dictionaries
        """
        if not self.is_trained:
            logger.warning("Model not trained. Returning default hold signal.")
            return [{'action': 'hold', 'confidence': 0.0, 'reason': 'Model not trained'}]
        
        try:
            # Prepare features
            X = self._prepare_features(data)
            
            # Get model prediction
            prediction = self.model.predict(X)[0]  # Single prediction

            return prediction
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return [{'action': 'hold', 'confidence': 0.0, 'reason': f'Prediction error: {str(e)}'}]
    
    
    def evaluate(self, data: Union[Dict, pd.DataFrame], target: Union[List, np.ndarray, pd.Series]) -> Dict:
        """
        Evaluate model performance on test data
        
        Parameters:
        - data: Test features
        - target: True target values
        
        Returns:
        - Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Prepare features
        X = self._prepare_features(data)
        y_true = np.array(target) if not isinstance(target, np.ndarray) else target
        
        # Get predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred),
            'n_samples': len(y_true),
            'prediction_range': {
                'min': float(y_pred.min()),
                'max': float(y_pred.max()),
                'mean': float(y_pred.mean()),
                'std': float(y_pred.std())
            }
        }
        
        # Add feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
            # Sort by importance
            metrics['feature_importance'] = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
        
        logger.info(f"Model evaluation completed. RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2_score']:.4f}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """
        Save the trained model to file
        
        Parameters:
        - filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model and metadata
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'training_metrics': self.training_metrics,
                'model_params': {
                    'iterations': self.iterations,
                    'learning_rate': self.learning_rate,
                    'depth': self.depth,
                    'l2_leaf_reg': self.l2_leaf_reg,
                    'random_seed': self.random_seed
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath: str):
        """
        Load a trained model from file
        
        Parameters:
        - filepath: Path to the saved model
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.training_metrics = model_data.get('training_metrics', {})
            
            # Update model parameters if available
            if 'model_params' in model_data:
                params = model_data['model_params']
                self.iterations = params.get('iterations', self.iterations)
                self.learning_rate = params.get('learning_rate', self.learning_rate)
                self.depth = params.get('depth', self.depth)
                self.l2_leaf_reg = params.get('l2_leaf_reg', self.l2_leaf_reg)
                self.random_seed = params.get('random_seed', self.random_seed)
            
            self.is_trained = True
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from trained model
        
        Returns:
        - Dictionary of feature names and their importance scores
        """
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return None
        
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def get_model_info(self) -> Dict:
        """
        Get information about the trained model
        
        Returns:
        - Dictionary with model information
        """
        info = {
            'is_trained': self.is_trained,
            'model_type': 'CatBoostRegressor',
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'model_params': {
                'iterations': self.iterations,
                'learning_rate': self.learning_rate,
                'depth': self.depth,
                'l2_leaf_reg': self.l2_leaf_reg,
                'random_seed': self.random_seed
            }
        }
        
        if self.is_trained:
            info['feature_importance'] = self.get_feature_importance()
        
        return info
