import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Model:
    def __init__(self):
        self.reg = LinearRegression()
        self.X_test = None
        self.y_test = None

    def train(self, X, y):
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Ensemble d'entraînement X : {X_train.shape}")
        print(f"Ensemble de test X : {self.X_test.shape}")
        print(f"Ensemble d'entraînement y : {y_train.shape}")
        print(f"Ensemble de test y : {self.y_test.shape}")
        
        self.reg.fit(X_train, y_train)
        return self.X_test

    def predict(self, X_test):
        return self.reg.predict(X_test)

    def evaluate(self, y_pred):
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = mse**0.5 
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print("RMSE:", rmse)
        print("MAE:", mae)
        print("R²:", r2)
