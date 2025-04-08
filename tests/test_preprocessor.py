import pandas as pd
import pytest
from src.preprocessor import Preprocessor

@pytest.fixture
def sample_df():
    data = {
        'price': [100000, 200000, 300000, 400000, 500000, 10000000], 
        'sqft_living': [1000, 1200, 1400, 1600, 1800, 2000],
        'bedrooms': [2, 3, 3, 4, 4, 5],
        'zipcode': ['98103', '98103', '98103', '98103', '98103', '98103'],
    }
    return pd.DataFrame(data)

def test_fit(sample_df):
    prep = Preprocessor()
    prep.fit(sample_df)
    assert prep.max_maison < sample_df['price'].max()

def test_transform(sample_df):
    prep = Preprocessor()
    prep.fit(sample_df)
    X, y = prep.transform(sample_df)
    
    assert 'price' not in X.columns
    assert y.name == 'price'
    assert all(X.dtypes != 'object')
    assert len(X) == 5

def test_fit_transform(sample_df):
    prep = Preprocessor()
    X, y = prep.fit_transform(sample_df)
    assert X.shape[0] == y.shape[0]
