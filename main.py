from src.preprocessor import Preprocessor
from src.model import Model
import pandas as pd

def main():
    df = pd.read_csv("./data/kc_house_data.csv")
    
    prep = Preprocessor()
    X, y = prep.fit_transform(df)
    
    model = Model()
    X_test = model.train(X, y)
    
    y_pred = model.predict(X_test)
    print("Prédictions :", y_pred[:5])
    
    model.evaluate(y_pred)

if __name__ == "__main__":
    main()
