import pandas as pd
import pytest
from src.model import Model

@pytest.fixture
def sample_data():
    data_size = 100
    feature1 = pd.Series([i / data_size for i in range(data_size)])
    feature2 = pd.Series([1 - (i / data_size) for i in range(data_size)])
    X = pd.DataFrame({'feature1': feature1, 'feature2': feature2})
    y = 2 * X['feature1'] + 3 * X['feature2']
    return X, y

def test_train(sample_data):
    X, y = sample_data
    model = Model()
    X_test = model.train(X, y)

    assert X_test.shape[1] == X.shape[1]
    assert len(model.y_test) == X_test.shape[0]

def test_predict(sample_data):
    X, y = sample_data
    model = Model()
    model.train(X, y)
    y_pred = model.predict(model.X_test)

    assert len(y_pred) == len(model.X_test)
    assert isinstance(y_pred, (list, pd.Series)) or hasattr(y_pred, '__len__')

def test_evaluate(sample_data):
    X, y = sample_data
    model = Model()
    model.train(X, y)
    y_pred = model.predict(model.X_test)

    try:
        model.evaluate(y_pred)
    except Exception as e:
        pytest.fail(f"La méthode evaluate a levé une exception : {e}")
