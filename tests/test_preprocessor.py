import pytest
import pandas as pd
import numpy as np
from src.preprocessor import HousePricePreprocessor

def test_preprocessor_init():
    """Test l'initialisation du préprocesseur."""
    prep = HousePricePreprocessor()
    assert prep.features == ['sqft_living']
    assert prep.target == 'price'

def test_preprocessor_transform():
    """Test la transformation des données."""
    # Création de données de test
    data = {
        'sqft_living': [1000, 2000, 3000],
        'price': [100000, 200000, 300000]
    }
    df = pd.DataFrame(data)
    
    # Test de la transformation
    prep = HousePricePreprocessor()
    X, y = prep.fit_transform(df)
    
    assert X.shape == (3, 1)
    assert y.shape == (3,)
    assert list(X.columns) == ['sqft_living']
    
def test_preprocessor_missing_columns():
    """Test la gestion des colonnes manquantes."""
    # Données sans la colonne price
    data = {'sqft_living': [1000, 2000, 3000]}
    df = pd.DataFrame(data)
    
    prep = HousePricePreprocessor()
    with pytest.raises(ValueError):
        prep.transform(df)

def test_preprocessor_handle_na():
    """Test la gestion des valeurs manquantes."""
    data = {
        'sqft_living': [1000, None, 3000],
        'price': [100000, 200000, 300000]
    }
    df = pd.DataFrame(data)
    
    prep = HousePricePreprocessor()
    X, y = prep.fit_transform(df)
    
    assert X.shape == (2, 1)  # Une ligne avec None devrait être supprimée