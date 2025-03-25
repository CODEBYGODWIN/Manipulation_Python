import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("kc_house_data.csv")  

plt.figure(figsize=(10, 5))
sns.scatterplot(x=df['bedrooms'], y=df['bathrooms'], alpha=0.5)
plt.title("Scatter plot of bedrooms and bathrooms")
plt.xlabel("bedrooms")
plt.ylabel("bathrooms")
plt.grid(True)


plt.savefig("scatter_plot.png")
