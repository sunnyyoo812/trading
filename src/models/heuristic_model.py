"""
Heuristic Model implementation that generates random predictions between -0.5% and 0.5%
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any

from .target_model import TargetModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeuristicModel(TargetModel):
    """
    Simple heuristic model that generates random predictions between -0.5% and 0.5%
    
    This model is useful for:
    - Baseline comparisons
    - Testing trading systems with random signals
    - Simulating market noise
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize HeuristicModel
        
        Parameters:
        - random_seed: Random seed for reproducible predictions
        """
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        self.is_trained = False
        self.training_data_size = 0
        
        logger.info("HeuristicModel initialized with random predictions between -0.5% and 0.5%")
    
    def predict(self, data: Union[Dict, pd.DataFrame]) -> float:
        """
        Generate a random prediction between -0.5% and 0.5%
        
        Parameters:
        - data: Input data (ignored, but kept for interface compatibility)
        
        Returns:
        - Random prediction between -0.5 and 0.5 (representing percentage change)
        """
        # Generate random value between -0.5 and 0.5
        prediction = self.rng.uniform(-0.5, 0.5)
        
        logger.debug(f"Heuristic prediction: {prediction:.4f}% price change")
        return float(prediction)
    
    def train(self, data: Union[Dict, pd.DataFrame], target: Union[List, np.ndarray, pd.Series]):
        """
        Mock training method - stores data size but doesn't actually train
        
        Parameters:
        - data: Training features (stored for reference but not used)
        - target: Target values (stored for reference but not used)
        """
        logger.info("Training HeuristicModel (mock training)...")
        
        # Convert data to consistent format for size calculation
        if isinstance(data, list):
            self.training_data_size = len(data)
        elif isinstance(data, pd.DataFrame):
            self.training_data_size = len(data)
        elif isinstance(data, dict):
            self.training_data_size = 1
        else:
            self.training_data_size = 0
        
        # Validate target size
        target_array = np.array(target) if not isinstance(target, np.ndarray) else target
        target_size = len(target_array)
        
        if self.training_data_size != target_size:
            logger.warning(f"Data and target size mismatch: {self.training_data_size} vs {target_size}")
        
        self.is_trained = True
        
        logger.info(f"HeuristicModel 'training' completed with {self.training_data_size} samples")
        logger.info("Note: This model generates random predictions regardless of training data")
    
    def evaluate(self, data: Union[Dict, pd.DataFrame], target: Union[List, np.ndarray, pd.Series]) -> Dict:
        """
        Evaluate the heuristic model by generating random predictions
        
        Parameters:
        - data: Test features
        - target: True target values
        
        Returns:
        - Dictionary of evaluation metrics
        """
        # Convert data to consistent format
        if isinstance(data, list):
            n_samples = len(data)
        elif isinstance(data, pd.DataFrame):
            n_samples = len(data)
        elif isinstance(data, dict):
            n_samples = 1
        else:
            n_samples = 0
        
        # Convert target to numpy array
        y_true = np.array(target) if not isinstance(target, np.ndarray) else target
        
        if len(y_true) != n_samples:
            raise ValueError(f"Data and target size mismatch: {n_samples} vs {len(y_true)}")
        
        # Generate random predictions for all samples
        y_pred = self.rng.uniform(-0.5, 0.5, size=n_samples)
        
        # Calculate basic metrics
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Calculate R² score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2_score = 1 - (ss_res / (ss_tot + 1e-8))  # Add small epsilon to avoid division by zero
        
        metrics = {
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2_score': r2_score,
            'n_samples': n_samples,
            'prediction_range': {
                'min': float(y_pred.min()),
                'max': float(y_pred.max()),
                'mean': float(y_pred.mean()),
                'std': float(y_pred.std())
            },
            'target_range': {
                'min': float(y_true.min()),
                'max': float(y_true.max()),
                'mean': float(y_true.mean()),
                'std': float(y_true.std())
            }
        }
        
        logger.info(f"Heuristic model evaluation completed. RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2_score']:.4f}")
        logger.info("Note: Performance metrics reflect random predictions vs actual targets")
        
        return metrics
    
    def save_model(self, filepath: str):
        """
        Save the heuristic model configuration
        
        Parameters:
        - filepath: Path to save the model
        """
        try:
            # Create directory if it doesn't exist (only if filepath has a directory)
            directory = os.path.dirname(filepath)
            if directory:  # Only create directory if filepath contains a directory path
                os.makedirs(directory, exist_ok=True)
            
            # Save model configuration
            model_data = {
                'model_type': 'HeuristicModel',
                'random_seed': self.random_seed,
                'is_trained': self.is_trained,
                'training_data_size': self.training_data_size,
                'prediction_range': [-0.5, 0.5]
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"HeuristicModel saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving HeuristicModel: {e}")
            raise
    
    def load_model(self, filepath: str):
        """
        Load a saved heuristic model configuration
        
        Parameters:
        - filepath: Path to the saved model
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Validate model type
            if model_data.get('model_type') != 'HeuristicModel':
                raise ValueError("Invalid model file - not a HeuristicModel")
            
            # Load configuration
            self.random_seed = model_data.get('random_seed', 42)
            self.is_trained = model_data.get('is_trained', False)
            self.training_data_size = model_data.get('training_data_size', 0)
            
            # Reinitialize random number generator
            self.rng = np.random.RandomState(self.random_seed)
            
            logger.info(f"HeuristicModel loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading HeuristicModel: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """
        Get information about the heuristic model
        
        Returns:
        - Dictionary with model information
        """
        return {
            'model_type': 'HeuristicModel',
            'is_trained': self.is_trained,
            'random_seed': self.random_seed,
            'training_data_size': self.training_data_size,
            'prediction_range': [-0.5, 0.5],
            'description': 'Generates random predictions between -0.5% and 0.5%'
        }
    
    def set_random_seed(self, seed: int):
        """
        Update the random seed for predictions
        
        Parameters:
        - seed: New random seed
        """
        self.random_seed = seed
        self.rng = np.random.RandomState(seed)
        logger.info(f"Random seed updated to {seed}")
    
    def generate_batch_predictions(self, n_predictions: int) -> np.ndarray:
        """
        Generate multiple random predictions at once
        
        Parameters:
        - n_predictions: Number of predictions to generate
        
        Returns:
        - Array of random predictions between -0.5 and 0.5
        """
        predictions = self.rng.uniform(-0.5, 0.5, size=n_predictions)
        logger.debug(f"Generated {n_predictions} batch predictions")
        return predictions
