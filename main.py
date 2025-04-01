"""
Main entry point for the Python data manipulation project.
"""
import os
import pandas as pd
from src.preprocessor import load_data, clean_data, get_feature_target, calculate_statistics, filter_data, percentage_of_data
from src.model import RegressionModel, plot_regression, train_test_model
from src.visualization import create_scatter_plot, create_histogram, create_correlation_heatmap, show_plot

def main():
    """
    Main function to run the complete data analysis pipeline.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    data_path = os.path.join(data_dir, "kc_house_data.csv")
    

    print("Loading data...")
    df = load_data(data_path)
    if df is None:
        return
    
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
  
    print("\nData Overview:")
    print(df.head())
    
   
    price_stats = calculate_statistics(df, "price")
    print("\nHouse Price Statistics:")
    for stat, value in price_stats.items():
        print(f"{stat.capitalize()}: {value:,.2f}")
    

    bedrooms_filter = df["bedrooms"] >= 3
    houses_3plus_bedrooms = filter_data(df, bedrooms_filter)
    percentage = percentage_of_data(houses_3plus_bedrooms, df)
    print(f"\nPercentage of houses with 3+ bedrooms: {percentage:.2f}%")
    
   
    print("\nCreating visualizations...")

    create_scatter_plot(
        df, 
        "sqft_living", 
        "price", 
        title="Relation between Living Area and House Price",
        xlabel="Living Area (sqft)",
        ylabel="Price ($)",
        save_path=os.path.join(data_dir, "scatter_plot.png")
    )
    print("Created scatter plot: data/scatter_plot.png")
    
 
    print("\nTraining regression model...")
    X, y = get_feature_target(df, ["sqft_living"], "price")
    
    model = RegressionModel()
    model.fit(X, y)
    
    r2 = model.score(X, y)
    print(f"Model R² score: {r2:.4f}")
    
  
    plot_regression(
        X, 
        y, 
        model.model, 
        save_path=os.path.join(data_dir, "regression.png")
    )
    print("Created regression plot: data/regression.png")
    

    print("\nEvaluating model with train-test split...")
    model, metrics, *_ = train_test_model(X, y)
    
    print("Training metrics:")
    for metric, value in metrics['train'].items():
        print(f"  {metric}: {value:.4f}")
    
    print("Testing metrics:")
    for metric, value in metrics['test'].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
