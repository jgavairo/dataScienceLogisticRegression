#!/usr/bin/env python3
"""
Prepare Test Data Script
Fills missing values and normalizes test data using training parameters.
"""

import pandas as pd
import numpy as np


def prepare_test_data(
    test_filepath='datasets/dataset_test.csv',
    params_filepath='datasets/dataset_train_filled_normalized_params.csv',
    output_filepath='datasets/dataset_test_prepared.csv'
):
    """
    Prepare test data with the same preprocessing as training data.
    
    Parameters:
    -----------
    test_filepath : str
        Path to raw test dataset
    params_filepath : str
        Path to normalization parameters from training
    output_filepath : str
        Path to save prepared test dataset
    """
    # Define the features we're using
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
    
    print("=" * 80)
    print("ORIGINAL TEST DATA")
    print("=" * 80)
    print(f"Shape: {df_test.shape}")
    print(f"\nMissing values:")
    print(df_test[selected_features].isnull().sum())
    
    # Load normalization parameters
    params_df = pd.read_csv(params_filepath, index_col=0)
    
    # Create a copy for processing
    df_prepared = df_test.copy()
    
    # Fill missing values and normalize
    print("\n" + "=" * 80)
    print("FILLING AND NORMALIZING TEST DATA")
    print("=" * 80)
    
    for feature in selected_features:
        if feature in params_df.index:
            mean_val = params_df.loc[feature, 'mean']
            std_val = params_df.loc[feature, 'std']
            
            # Count missing values
            missing_count = df_prepared[feature].isnull().sum()
            
            # Fill missing values with training mean
            df_prepared[feature].fillna(mean_val, inplace=True)
            
            # Normalize with training parameters
            df_prepared[feature] = (df_prepared[feature] - mean_val) / std_val
            
            print(f"✓ {feature}:")
            print(f"    Filled {missing_count} missing values")
            print(f"    Normalized (mean={mean_val:.4f}, std={std_val:.4f})")
    
    # Verify no missing values remain
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    print(f"Missing values after processing:")
    print(df_prepared[selected_features].isnull().sum())
    print(f"\nTotal missing values: {df_prepared[selected_features].isnull().sum().sum()}")
    
    # Save prepared data
    df_prepared.to_csv(output_filepath, index=False)
    
    print("\n" + "=" * 80)
    print("TEST DATA PREPARED")
    print("=" * 80)
    print(f"✓ Test data saved to: {output_filepath}")
    print(f"\nFirst 5 rows of normalized features:")
    print(df_prepared[selected_features].head())
    
    return df_prepared


def main():
    """Main function to prepare test data."""
    df_prepared = prepare_test_data()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✓ Missing values filled with training means")
    print("✓ Features normalized with training parameters")
    print("✓ Ready for prediction!")


if __name__ == "__main__":
    main()
