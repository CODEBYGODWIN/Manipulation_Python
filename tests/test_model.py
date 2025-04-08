import pytest
import pandas as pd
import numpy as np
from src.model import HousePriceModel
import os

def test_model_init():
    """Test l'initialisation du modèle."""
    model = HousePriceModel()
    assert hasattr(model, 'model')

def test_model_train_predict():
    """Test l'entraînement et la prédiction du modèle."""
    # Création de données de test
    X = pd.DataFrame({'sqft_living': [1000, 2000, 3000]})
    y = pd.Series([100000, 200000, 300000])
    
    # Entraînement du modèle
    model = HousePriceModel()
    model.train(X, y)
    
    # Test des prédictions
    predictions = model.predict(X)
    assert len(predictions) == 3
    
    # Test du score
    score = model.score(X, y)
    assert 0 <= score <= 1

def test_model_plot():
    """Test la génération du graphique."""
    # Création de données de test
    X = pd.DataFrame({'sqft_living': [1000, 2000, 3000]})
    y = pd.Series([100000, 200000, 300000])
    
    # Test du plot
    model = HousePriceModel()
    model.train(X, y)
    
    test_plot = "test_regression.png"
    model.plot_regression(X, y, save_path=test_plot)
    
    # Vérifie que le fichier a été créé
    assert os.path.exists(test_plot)
    
    # Nettoyage
    os.remove(test_plot)