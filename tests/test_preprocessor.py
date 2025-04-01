"""
Tests for the preprocessor module.
"""
import unittest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessor import (
    load_data, 
    clean_data, 
    get_feature_target, 
    calculate_statistics,
    filter_data,
    percentage_of_data
)

class TestPreprocessor(unittest.TestCase):
    """Test cases for the preprocessor module."""
    
    def setUp(self):
        """Set up test data."""
 
        self.df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': [1.1, 2.2, np.nan, 4.4, 5.5]
        })
        
    def test_clean_data(self):
        """Test the clean_data function."""
        cleaned_df = clean_data(self.df)
        self.assertEqual(len(cleaned_df), 4)  
        self.assertFalse(cleaned_df.isna().any().any())
        
    def test_get_feature_target(self):
        """Test the get_feature_target function."""
        X, y = get_feature_target(self.df, ['A', 'B'], 'C')
        self.assertEqual(X.shape, (5, 2))
        self.assertEqual(y.shape, (5,))
        self.assertTrue(all(X.columns == ['A', 'B']))
        
    def test_calculate_statistics(self):
        """Test the calculate_statistics function."""
        stats = calculate_statistics(self.df, 'A')
        self.assertEqual(stats['mean'], 3)
        self.assertEqual(stats['median'], 3)
        self.assertEqual(stats['min'], 1)
        self.assertEqual(stats['max'], 5)
        
    def test_filter_data(self):
        """Test the filter_data function."""
        filtered = filter_data(self.df, self.df['A'] > 3)
        self.assertEqual(len(filtered), 2)
        self.assertTrue(all(filtered['A'] > 3))
        
    def test_percentage_of_data(self):
        """Test the percentage_of_data function."""
        filtered = filter_data(self.df, self.df['A'] > 3)
        percentage = percentage_of_data(filtered, self.df)
        self.assertEqual(percentage, 40.0)  

if __name__ == '__main__':
    unittest.main()
