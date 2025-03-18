import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("kc_house_data.csv")  

plt.figure(figsize=(10, 5))
sns.scatterplot(x=df['sqft_living'], y=df['price'], alpha=0.5)
plt.xlabel("Surface habitable (sqft)")
plt.ylabel("Prix des maisons")
plt.title("Relation entre la surface et le prix des maisons")
plt.show()
