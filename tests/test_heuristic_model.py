#!/usr/bin/env python3
"""
Test script for HeuristicModel
"""

import numpy as np
import pandas as pd
from src.models.heuristic_model import HeuristicModel

def test_heuristic_model():
    """Test the HeuristicModel functionality"""
    print("Testing HeuristicModel...")
    
    # Initialize model
    model = HeuristicModel(random_seed=42)
    print(f"Model info: {model.get_model_info()}")
    
    # Test single prediction
    test_data = {
        'bid_price': 100.0,
        'ask_price': 100.1,
        'volume': 1000
    }
    
    prediction = model.predict(test_data)
    print(f"Single prediction: {prediction:.4f}%")
    
    # Test multiple predictions
    predictions = []
    for i in range(10):
        pred = model.predict(test_data)
        predictions.append(pred)
    
    print(f"10 predictions: {[f'{p:.4f}' for p in predictions]}")
    print(f"Prediction range: {min(predictions):.4f} to {max(predictions):.4f}")
    
    # Test batch predictions
    batch_preds = model.generate_batch_predictions(5)
    print(f"Batch predictions: {[f'{p:.4f}' for p in batch_preds]}")
    
    # Test training (mock)
    training_data = [
        {'price': 100.0, 'volume': 1000},
        {'price': 101.0, 'volume': 1100},
        {'price': 99.5, 'volume': 900}
    ]
    training_targets = [0.1, -0.2, 0.3]
    
    model.train(training_data, training_targets)
    print(f"Model after training: {model.get_model_info()}")
    
    # Test evaluation
    test_targets = [0.1, -0.1, 0.2, -0.3, 0.0]
    test_data_list = [test_data] * 5
    
    metrics = model.evaluate(test_data_list, test_targets)
    print(f"Evaluation metrics:")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v:.4f}")
        else:
            print(f"  {key}: {value:.4f}")
    
    # Test save/load
    model.save_model("test_heuristic_model.pkl")
    
    # Load model
    new_model = HeuristicModel()
    new_model.load_model("test_heuristic_model.pkl")
    
    # Test that loaded model works
    new_prediction = new_model.predict(test_data)
    print(f"Prediction from loaded model: {new_prediction:.4f}%")
    
    print("HeuristicModel test completed successfully!")

if __name__ == "__main__":
    test_heuristic_model()
