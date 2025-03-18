from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

def linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict(model, X):
    return model.predict(X)

def score(model, X, y):
    return model.score(X, y)

def plot_regression(X, y, model):
    plt.figure(figsize=(10, 5))
    
    
    plt.scatter(X, y, alpha=0.5, label="Données réelles")
    
    
    plt.plot(X, model.predict(X), color='red', label="Régression linéaire")
    
    plt.xlabel("Surface habitable (sqft)")
    plt.ylabel("Prix des maisons")
    plt.title("Régression linéaire simple")
    plt.legend()
    plt.savefig("regression.png")


def main():
    try:
        df = pd.read_csv("kc_house_data.csv")
        
       
        if 'sqft_living' not in df.columns or 'price' not in df.columns:
            raise KeyError("Les colonnes nécessaires ne sont pas dans le fichier CSV.")
        
       
        df = df.dropna(subset=['sqft_living', 'price'])

        
        X = df[['sqft_living']]
        y = df['price']

        
        model = linear_regression(X, y)

        
        print(f"Score R² du modèle : {score(model, X, y):.4f}")

        
        plot_regression(X, y, model)

    except FileNotFoundError:
        print("Erreur : Le fichier 'kc_house_data.csv' est introuvable.")
    except KeyError as e:
        print(f"Erreur : {e}")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")

if __name__ == "__main__":
    main()
