import pandas as pd

class Preprocessor:
    def __init__(self):
        self.max_maison = None

    def fit(self, df):
        self.max_maison = df['price'].quantile(0.99)

    def transform(self, df):
        df = df[df['price'] < self.max_maison]
        df = df.select_dtypes(include=['float64', 'int64'])

        X = df.drop('price', axis=1)
        y = df['price']

        return X, y

    def fit_transform(self, df):
        self.fit(df)
        X, y = self.transform(df)
        return X, y
