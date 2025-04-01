"""
Data preprocessing module for handling data cleaning, transformation, and preparation.
"""
import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

def clean_data(df):
    """
    Clean the dataframe by removing missing values.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    if df is None:
        return None
    return df.dropna()

def get_feature_target(df, feature_cols, target_col):
    """
    Split dataframe into features and target.
    
    Args:
        df (pd.DataFrame): Input dataframe
        feature_cols (list): List of feature column names
        target_col (str): Target column name
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector
    """
    if df is None:
        return None, None
    
    try:
        X = df[feature_cols]
        y = df[target_col]
        return X, y
    except KeyError as e:
        print(f"Error: {e}")
        return None, None

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name
        
    Returns:
        dict: Dictionary of statistics
    """
    if df is None or column not in df.columns:
        return None
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max()
    }
    
    return stats

def filter_data(df, condition):
    """
    Filter dataframe based on a condition.
    
    Args:
        df (pd.DataFrame): Input dataframe
        condition: Boolean condition for filtering
        
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    if df is None:
        return None
    
    return df[condition]

def percentage_of_data(filtered_df, original_df):
    """
    Calculate what percentage the filtered dataframe is of the original.
    
    Args:
        filtered_df (pd.DataFrame): Filtered dataframe
        original_df (pd.DataFrame): Original dataframe
        
    Returns:
        float: Percentage
    """
    if filtered_df is None or original_df is None:
        return None
    
    return (len(filtered_df) / len(original_df)) * 100
