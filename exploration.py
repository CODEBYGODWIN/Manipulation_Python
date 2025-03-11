import pandas as pd

df = pd.read_csv("kc_house_data.csv")

print(df.head())

print("Moyenne des prix : ", df["price"].mean())

maisons_3_chambres_plus = df[df["bedrooms"] >= 3]
pourcentage_maisons_3_chambres_plus = (len(maisons_3_chambres_plus) / len(df)) * 100

print(f"Pourcentage de maisons avec 3 chambres ou plus : {pourcentage_maisons_3_chambres_plus}%")
