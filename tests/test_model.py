"""
Tests for the model module.
"""
import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add the parent directory to the path so we can import the src module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import RegressionModel, train_test_model

class TestModel(unittest.TestCase):
    """Test cases for the model module."""
    
    def setUp(self):
        """Set up test data."""
    
        np.random.seed(42)
        X = np.random.rand(100, 1)
        y = 2 * X.flatten() + 1 + 0.1 * np.random.randn(100)
        self.X = X
        self.y = y
        
    def test_regression_model_init(self):
        """Test RegressionModel initialization."""
        model = RegressionModel()
        self.assertFalse(model.is_fitted)
        
    
        with self.assertRaises(ValueError):
            RegressionModel(model_type='invalid_type')
        
    def test_regression_model_fit_predict(self):
        """Test RegressionModel fit and predict methods."""
        model = RegressionModel()
        model.fit(self.X, self.y)
        
        self.assertTrue(model.is_fitted)
        
       
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))
        
     
        coef = model.model.coef_[0]
        intercept = model.model.intercept_
        self.assertAlmostEqual(coef, 2.0, delta=0.3)
        self.assertAlmostEqual(intercept, 1.0, delta=0.3)
        
    def test_regression_model_score(self):
        """Test RegressionModel score method."""
        model = RegressionModel()
        model.fit(self.X, self.y)
        
        score = model.score(self.X, self.y)
        self.assertGreater(score, 0.9)  
        
    def test_regression_model_evaluate(self):
        """Test RegressionModel evaluate method."""
        model = RegressionModel()
        model.fit(self.X, self.y)
        
        metrics = model.evaluate(self.X, self.y)
        
        self.assertIn('r2', metrics)
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        
        self.assertGreater(metrics['r2'], 0.9)
        self.assertLess(metrics['mse'], 0.1)
        self.assertLess(metrics['rmse'], 0.3)
        
    def test_train_test_model(self):
        """Test train_test_model function."""
        model, metrics, X_train, X_test, y_train, y_test = train_test_model(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
     
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        

        self.assertIn('train', metrics)
        self.assertIn('test', metrics)
        
     
        for dataset in ['train', 'test']:
            self.assertIn('r2', metrics[dataset])
            self.assertIn('mse', metrics[dataset])
            self.assertIn('rmse', metrics[dataset])

if __name__ == '__main__':
    unittest.main()
