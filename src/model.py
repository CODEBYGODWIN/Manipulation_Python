from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class HousePriceModel:
    """Modèle de prédiction des prix des maisons."""
    
    def __init__(self):
        self.model = LinearRegression()
        
    def train(self, X, y):
        """Entraîne le modèle sur les données."""
        self.model.fit(X, y)
        return self
        
    def predict(self, X):
        """Fait des prédictions sur de nouvelles données."""
        return self.model.predict(X)
        
    def score(self, X, y):
        """Calcule le score R² du modèle."""
        return self.model.score(X, y)
        
    def plot_regression(self, X, y, save_path="regression.png"):
        """Trace la régression linéaire et les données."""
        plt.figure(figsize=(10, 5))
        
        # Données réelles
        plt.scatter(X, y, alpha=0.5, label="Données réelles")
        
        # Ligne de régression
        plt.plot(X, self.predict(X), color='red', label="Régression linéaire")
        
        plt.xlabel("Surface habitable (sqft)")
        plt.ylabel("Prix des maisons")
        plt.title("Régression linéaire simple")
        plt.legend()
        
        # Sauvegarde du graphique
        plt.savefig(save_path)
        plt.close()