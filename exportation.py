import pandas as pd

df = pd.read_csv('kc_house_data.csv')

print(df.head())

maisons_3_plus = df[df["bedrooms"] >= 3]

pourcentage = (len(maisons_3_plus) / len(df)) * 100

print(f"Pourcentage des maisons avec 3 chambres ou plus : {pourcentage}%")

from ydata_profiling import ProfileReport

profile = ProfileReport(df, title="Profiling Report")
profile.to_file("your_report.html")
