#!/usr/bin/env python3
"""
Preprocessing utilities for the logistic regression model.
Contains functions for data loading, cleaning, and normalization.
"""

import pandas as pd
import numpy as np


def load_normalization_params(filepath='output/normalization_params.csv'):
    """
    Load normalization parameters (mean and std) from training.
    
    Parameters:
    -----------
    filepath : str
        Path to the normalization parameters CSV file
        
    Returns:
    --------
    dict
        Dictionary with feature names as keys and {'mean', 'std'} as values
    """
    params_df = pd.read_csv(filepath, index_col=0)
    params_dict = {}
    
    for feature in params_df.index:
        params_dict[feature] = {
            'mean': params_df.loc[feature, 'mean'],
            'std': params_df.loc[feature, 'std']
        }
    
    return params_dict


def clean_and_normalize(df, selected_features, normalization_params):
    """
    Clean data and normalize using the same parameters as training.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe
    selected_features : list
        List of feature names to extract
    normalization_params : dict
        Dictionary containing mean and std for each feature
        
    Returns:
    --------
    np.ndarray
        Cleaned and normalized feature matrix
    """
    # Extract selected features
    X = df[selected_features].copy()
    
    # Fill missing values with column means (using training means)
    for feature in selected_features:
        if feature in normalization_params:
            # Fill NaN with the training mean
            X[feature].fillna(normalization_params[feature]['mean'], inplace=True)
    
    # Normalize using training parameters
    X_normalized = X.copy()
    for feature in selected_features:
        if feature in normalization_params:
            mean_val = normalization_params[feature]['mean']
            std_val = normalization_params[feature]['std']
            X_normalized[feature] = (X[feature] - mean_val) / std_val
    
    return X_normalized.values


def add_bias(X):
    """
    Add bias column (column of ones) to the feature matrix.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
        
    Returns:
    --------
    np.ndarray
        Feature matrix with bias column added as first column
    """
    n_samples = X.shape[0]
    bias_column = np.ones((n_samples, 1))
    return np.hstack((bias_column, X))


def load_and_preprocess_test_data(
    test_filepath='datasets/dataset_test.csv',
    params_filepath='output/normalization_params.csv'
):
    """
    Load and preprocess test data with the same pipeline as training.
    
    Parameters:
    -----------
    test_filepath : str
        Path to test dataset
    params_filepath : str
        Path to normalization parameters
        
    Returns:
    --------
    X_test : np.ndarray
        Preprocessed test feature matrix with bias
    original_df : pd.DataFrame
        Original dataframe (for keeping Index, names, etc.)
    """
    # Define the selected features (same as training)
    selected_features = [
        'Astronomy',
        'Herbology',
        'Divination',
        'Muggle Studies',
        'Ancient Runes',
        'History of Magic',
        'Transfiguration',
        'Potions',
        'Charms',
        'Flying'
    ]
    
    # Load test data
    df_test = pd.read_csv(test_filepath)
    
    # Load normalization parameters from training
    normalization_params = load_normalization_params(params_filepath)
    
    # Clean and normalize
    X_test = clean_and_normalize(df_test, selected_features, normalization_params)
    
    # Add bias
    X_test = add_bias(X_test)
    
    print("=" * 80)
    print("TEST DATA PREPROCESSING")
    print("=" * 80)
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features (with bias): {X_test.shape[1]}")
    print(f"✓ Missing values filled with training means")
    print(f"✓ Data normalized with training parameters")
    print(f"✓ Bias column added")
    
    return X_test, df_test


if __name__ == "__main__":
    # Test the preprocessing
    X_test, df_test = load_and_preprocess_test_data()
    print(f"\nFirst 5 rows of preprocessed test data:")
    print(X_test[:5])
