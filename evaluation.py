import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv("kc_house_data.csv")

df = df.select_dtypes(include=['float64', 'int64'])

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Ensemble d'entraînement X : {X_train.shape}")
print(f"Ensemble de test X : {X_test.shape}")
print(f"Ensemble d'entraînement y : {y_train.shape}")
print(f"Ensemble de test y : {y_test.shape}")

reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

print("Prédictions des prix : ", y_pred[:5])
print("vrai prix : ", y_test[:5])

mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5 
mae = mean_absolute_error(y_test, y_pred)

print("RMSE:", rmse)
