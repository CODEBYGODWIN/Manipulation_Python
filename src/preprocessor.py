import pandas as pd
import numpy as np

class HousePricePreprocessor:
    """Préprocesseur pour les données de prix des maisons."""
    
    def fit(self, df):
        """Apprend les paramètres de prétraitement sur les données."""
        # Pour l'instant, pas de paramètres à apprendre
        return self
        
    def transform(self, df):
        """Transforme les données en features X et target y."""
        # Vérification des colonnes
        missing_cols = set(self.features + [self.target]) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Colonnes manquantes: {missing_cols}")
            
        # Nettoyage des données
        df = df.dropna(subset=self.features + [self.target])
        
        # Extraction des features et target
        X = df[self.features]
        y = df[self.target]
        
        return X, y
        
    def fit_transform(self, df):
        """Combine fit et transform en une seule opération."""
        return self.fit(df).transform(df)