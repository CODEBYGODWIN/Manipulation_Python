import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("kc_house_data.csv")

missing_values = df.isnull().sum()
print("Nombre de valeurs manquantes par colonne :\n", missing_values)

missing_percentage = (df.isnull().sum() / len(df)) * 100
print("Pourcentage de valeurs manquantes par colonne :\n", missing_percentage)

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title("Carte des valeurs manquantes")
plt.show()