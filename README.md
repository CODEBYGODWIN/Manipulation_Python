# Projet d'Analyse des Prix Immobiliers

Ce projet implémente une analyse simple des prix immobiliers en utilisant la régression linéaire.

## Structure du Projet

```
.
├── README.md
├── pyproject.toml
├── .gitignore
├── main.py
├── data/
│   └── kc_house_data.csv
├── src/
│   ├── __init__.py
│   ├── preprocessor.py
│   └── model.py
└── tests/
    ├── __init__.py
    ├── test_preprocessor.py
    └── test_model.py
```

## Installation

1. Assurez-vous d'avoir Python 3.8+ installé
2. Installez poetry (gestionnaire de dépendances)
3. Installez les dépendances :
   ```bash
   poetry install
   ```

## Utilisation

1. Placez votre fichier de données `kc_house_data.csv` dans le dossier `data/`
2. Exécutez le programme :
   ```bash
   python main.py
   ```

## Tests

Pour lancer les tests unitaires :
```bash
pytest
```

## Structure du Code

- `src/preprocessor.py` : Contient la classe `HousePricePreprocessor` pour le prétraitement des données
- `src/model.py` : Contient la classe `HousePriceModel` pour la régression linéaire
- `tests/` : Contient les tests unitaires