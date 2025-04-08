import pandas as pd
from src.preprocessor import HousePricePreprocessor
from src.model import HousePriceModel

def main():
    """Point d'entrée principal du projet."""
    try:
        # Chargement des données
        print("Chargement des données...")
        df = pd.read_csv("data/kc_house_data.csv")
        
        # Prétraitement
        print("Prétraitement des données...")
        preprocessor = HousePricePreprocessor()
        X, y = preprocessor.fit_transform(df)
        
        # Entraînement du modèle
        print("Entraînement du modèle...")
        model = HousePriceModel()
        model.train(X, y)
        
        # Évaluation
        print("\nRésultats:")
        print(f"Score R² du modèle: {model.score(X, y):.4f}")
        
        # Visualisation
        print("\nGénération du graphique de régression...")
        model.plot_regression(X.iloc[:, 0], y)
        print("Graphique sauvegardé dans 'regression.png'")
        
    except FileNotFoundError:
        print("Erreur: Fichier de données non trouvé. Assurez-vous que 'kc_house_data.csv' est présent dans le dossier 'data/'")
    except Exception as e:
        print(f"Erreur: {str(e)}")

if __name__ == "__main__":
    main()