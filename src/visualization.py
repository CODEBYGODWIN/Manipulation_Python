"""
Visualization module for creating plots and charts.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def create_scatter_plot(df, x_col, y_col, title=None, xlabel=None, ylabel=None, save_path=None):
    """
    Create a scatter plot.
    
    Args:
        df (pd.DataFrame): Input dataframe
        x_col (str): Column name for x-axis
        y_col (str): Column name for y-axis
        title (str, optional): Plot title
        xlabel (str, optional): X-axis label
        ylabel (str, optional): Y-axis label
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    plt.figure(figsize=(10, 5))
    
    sns.scatterplot(x=df[x_col], y=df[y_col], alpha=0.5)
    
    plt.xlabel(xlabel or x_col)
    plt.ylabel(ylabel or y_col)
    plt.title(title or f"Relation entre {x_col} et {y_col}")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        return None
    
    return plt.gcf()

def create_histogram(df, col, bins=30, title=None, xlabel=None, ylabel="Fréquence", save_path=None):
    """
    Create a histogram.
    
    Args:
        df (pd.DataFrame): Input dataframe
        col (str): Column name to plot
        bins (int, optional): Number of bins
        title (str, optional): Plot title
        xlabel (str, optional): X-axis label
        ylabel (str, optional): Y-axis label
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    plt.figure(figsize=(10, 5))
    
    sns.histplot(df[col], bins=bins, kde=True)
    
    plt.xlabel(xlabel or col)
    plt.ylabel(ylabel)
    plt.title(title or f"Distribution de {col}")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        return None
    
    return plt.gcf()

def create_correlation_heatmap(df, columns=None, title="Matrice de corrélation", save_path=None):
    """
    Create a correlation heatmap.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list, optional): List of columns to include
        title (str, optional): Plot title
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    if columns:
        corr_df = df[columns].corr()
    else:
        corr_df = df.select_dtypes(include=[np.number]).corr()
    
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        return None
    
    return plt.gcf()

def create_boxplot(df, x_col, y_col=None, title=None, xlabel=None, ylabel=None, save_path=None):
    """
    Create a boxplot.
    
    Args:
        df (pd.DataFrame): Input dataframe
        x_col (str): Column name for x-axis
        y_col (str, optional): Column name for y-axis
        title (str, optional): Plot title
        xlabel (str, optional): X-axis label
        ylabel (str, optional): Y-axis label
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    plt.figure(figsize=(10, 5))
    
    if y_col:
        sns.boxplot(x=df[x_col], y=df[y_col])
        plt.ylabel(ylabel or y_col)
    else:
        sns.boxplot(x=df[x_col])
    
    plt.xlabel(xlabel or x_col)
    plt.title(title or f"Boxplot de {x_col}")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        return None
    
    return plt.gcf()

def show_plot():
    """
    Display the current plot.
    """
    plt.show()
