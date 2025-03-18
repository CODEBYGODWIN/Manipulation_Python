import pandas as pd

df = pd.read_csv("data/kc_house_data.csv")

print(df.head())

# % de maison >= 3 chambres

from ydata_profiling import ProfileReport

profile = ProfileReport(df, title="Profiling Report")
profile.to_file("your_report.html")
#print(df["price"].mean())