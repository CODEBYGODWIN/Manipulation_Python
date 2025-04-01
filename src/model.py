"""
Model module for training, evaluating, and using machine learning models.
"""
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class RegressionModel:
    """
    A class to handle regression modeling tasks.
    """
    def __init__(self, model_type='linear'):
        """
        Initialize the regression model.
        
        Args:
            model_type (str): Type of regression model to use
        """
        if model_type == 'linear':
            self.model = LinearRegression()
        else:
            raise ValueError(f"Model type '{model_type}' not supported")
        
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Fit the model to the training data.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            self: The fitted model
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted model.
        
        Args:
            X: Features
            
        Returns:
            array: Predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def score(self, X, y):
        """
        Calculate the R² score of the model.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            float: R² score
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        
        return self.model.score(X, y)
    
    def evaluate(self, X, y):
        """
        Evaluate the model with multiple metrics.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")
        
        y_pred = self.predict(X)
        
        metrics = {
            'r2': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred))
        }
        
        return metrics

def plot_regression(X, y, model, save_path=None):
    """
    Plot regression results.
    
    Args:
        X: Features (should be 1D or convertible to 1D)
        y: Target values
        model: Fitted model object with predict method
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 5))
    
    # Convert X to 1D array for plotting if it's not already
    X_plot = X.flatten() if hasattr(X, 'flatten') else X
    if hasattr(X, 'values') and X.shape[1] == 1:
        X_plot = X.values.flatten()
    
    # Sort X and corresponding predictions for a clean line plot
    sort_idx = np.argsort(X_plot)
    X_sorted = X_plot[sort_idx]
    
    # Ensure X is in the right format for prediction
    X_pred = X_sorted.reshape(-1, 1) if hasattr(X_sorted, 'reshape') else X
    
    # Make predictions
    y_pred = model.predict(X_pred)
    
    # Plot
    plt.scatter(X_plot, y, alpha=0.5, label="Données réelles")
    plt.plot(X_sorted, y_pred[sort_idx] if hasattr(X_pred, 'reshape') else y_pred, 
             color='red', label="Régression linéaire")
    
    plt.xlabel("Variable indépendante")
    plt.ylabel("Variable dépendante")
    plt.title("Régression linéaire")
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def train_test_model(X, y, test_size=0.2, random_state=42):
    """
    Train a model with train-test split.
    
    Args:
        X: Features
        y: Target
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (model, metrics, X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    model = RegressionModel()
    model.fit(X_train, y_train)
    
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)
    
    metrics = {
        'train': train_metrics,
        'test': test_metrics
    }
    
    return model, metrics, X_train, X_test, y_train, y_test
