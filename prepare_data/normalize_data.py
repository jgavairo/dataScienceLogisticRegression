#!/usr/bin/env python3
"""
Normalize Data Script
Normalizes all float features using standardization (z-score normalization).
Formula: z = (x - mean) / std
"""

import pandas as pd
import numpy as np


def normalize_data(filepath='datasets/dataset_train_filled.csv', output_path=None):
    """
    Normalize all float64 features using z-score normalization.
    
    Parameters:
    -----------
    filepath : str
        Path to the filled dataset CSV file
    output_path : str, optional
        Path to save the normalized dataset. If None, uses default naming.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with normalized float features
    dict
        Dictionary containing mean and std for each normalized column
    """
    # Load the dataset
    df = pd.read_csv(filepath)
    
    print("=" * 80)
    print("ORIGINAL DATASET INFO")
    print("=" * 80)
    print(f"Dataset shape: {df.shape}")
    print(f"\nHogwarts House count:")
    print(df['Hogwarts House'].value_counts())
    
    # Create a copy to avoid modifying original
    df_normalized = df.copy()
    
    # Get all float64 columns (excluding Hogwarts House)
    float_columns = df_normalized.select_dtypes(include=['float64']).columns.tolist()
    
    print("\n" + "=" * 80)
    print("FLOAT64 COLUMNS TO NORMALIZE")
    print("=" * 80)
    print(f"Columns: {float_columns}")
    print(f"Number of float columns: {len(float_columns)}")
    
    # Dictionary to store normalization parameters (mean and std)
    normalization_params = {}
    
    print("\n" + "=" * 80)
    print("NORMALIZING DATA (Z-SCORE NORMALIZATION)")
    print("=" * 80)
    
    # Normalize each float column
    for column in float_columns:
        mean_value = df_normalized[column].mean()
        std_value = df_normalized[column].std()
        
        # Store parameters for potential denormalization later
        normalization_params[column] = {
            'mean': mean_value,
            'std': std_value
        }
        
        # Apply z-score normalization: z = (x - mean) / std
        df_normalized[column] = (df_normalized[column] - mean_value) / std_value
        
        print(f"✓ {column}:")
        print(f"  Mean: {mean_value:.6f}, Std: {std_value:.6f}")
        print(f"  New range: [{df_normalized[column].min():.6f}, {df_normalized[column].max():.6f}]")
    
    print("\n" + "=" * 80)
    print("NORMALIZED DATA STATISTICS")
    print("=" * 80)
    print(df_normalized[float_columns].describe())
    
    print("\n" + "=" * 80)
    print("VERIFICATION: MEAN AND STD OF NORMALIZED DATA")
    print("=" * 80)
    for column in float_columns:
        print(f"{column}:")
        print(f"  Mean: {df_normalized[column].mean():.10f} (should be ~0)")
        print(f"  Std:  {df_normalized[column].std():.10f} (should be ~1)")
    
    print("\n" + "=" * 80)
    print("HOGWARTS HOUSE COUNT VERIFICATION")
    print("=" * 80)
    print(df_normalized['Hogwarts House'].value_counts())
    
    # Save the normalized dataset
    if output_path is None:
        output_path = filepath.replace('.csv', '_normalized.csv')
        if output_path == filepath:  # If no '_filled' in filename
            output_path = filepath.replace('.csv', '_normalized.csv')
    
    df_normalized.to_csv(output_path, index=False)
    print(f"\n✓ Normalized dataset saved to: {output_path}")
    
    # Save normalization parameters for future use
    params_path = output_path.replace('.csv', '_params.csv')
    params_df = pd.DataFrame(normalization_params).T
    params_df.to_csv(params_path)
    print(f"✓ Normalization parameters saved to: {params_path}")
    
    return df_normalized, normalization_params


def main():
    """Main function to normalize data."""
    # Normalize the filled dataset
    df_normalized, params = normalize_data()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Dataset shape: {df_normalized.shape}")
    print(f"\nFirst 5 rows of normalized dataset:")
    print(df_normalized.head())
    
    print("\n" + "=" * 80)
    print("NORMALIZATION PARAMETERS")
    print("=" * 80)
    for col, values in params.items():
        print(f"{col}: mean={values['mean']:.6f}, std={values['std']:.6f}")


if __name__ == "__main__":
    main()
