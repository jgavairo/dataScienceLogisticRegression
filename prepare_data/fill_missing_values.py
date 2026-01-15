import pandas as pd
import numpy as np


def fill_missing_values(filepath='datasets/dataset_train.csv', output_path=None):
    """
    Fill missing values in numeric features with column means.
    
    Parameters:
    -----------
    filepath : str
        Path to the dataset CSV file
    output_path : str, optional
        Path to save the filled dataset. If None, uses default naming.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with missing values filled
    """
    # Load the dataset
    df = pd.read_csv(filepath)
    
    print("=" * 80)
    print("ORIGINAL DATASET INFO")
    print("=" * 80)
    print(f"Dataset shape: {df.shape}")
    print(f"\nHogwarts House count:")
    print(df['Hogwarts House'].value_counts())
    
    print("\n" + "=" * 80)
    print("MISSING VALUES BEFORE FILLING")
    print("=" * 80)
    missing_before = df.isnull().sum()
    print(missing_before)
    print(f"\nTotal missing values: {missing_before.sum()}")
    
    # Create a copy to avoid modifying original
    df_filled = df.copy()
    
    # Get all float64 columns (excluding Hogwarts House)
    float_columns = df_filled.select_dtypes(include=['float64']).columns.tolist()
    
    print("\n" + "=" * 80)
    print("FILLING MISSING VALUES WITH COLUMN MEANS")
    print("=" * 80)
    print(f"Float64 columns to process: {float_columns}")
    
    # Fill missing values with mean for each float column
    for column in float_columns:
        if df_filled[column].isnull().sum() > 0:
            mean_value = df_filled[column].mean()
            missing_count = df_filled[column].isnull().sum()
            df_filled[column].fillna(mean_value, inplace=True)
            print(f"✓ {column}: Filled {missing_count} missing values with mean = {mean_value:.6f}")
        else:
            print(f"  {column}: No missing values")
    
    print("\n" + "=" * 80)
    print("MISSING VALUES AFTER FILLING")
    print("=" * 80)
    missing_after = df_filled.isnull().sum()
    print(missing_after)
    print(f"\nTotal missing values: {missing_after.sum()}")
    
    print("\n" + "=" * 80)
    print("HOGWARTS HOUSE COUNT VERIFICATION")
    print("=" * 80)
    print(df_filled['Hogwarts House'].value_counts())
    print(f"\nHogwarts House has {df_filled['Hogwarts House'].isnull().sum()} missing values")
    
    # Save the filled dataset
    if output_path is None:
        output_path = filepath.replace('.csv', '_filled.csv')
    
    df_filled.to_csv(output_path, index=False)
    print(f"\n✓ Filled dataset saved to: {output_path}")
    
    return df_filled


def main():
    """Main function to fill missing values."""
    # Fill missing values in the selected dataset
    df_filled = fill_missing_values()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Dataset shape: {df_filled.shape}")
    print(f"Data types:\n{df_filled.dtypes}")
    print(f"\nFirst 5 rows of filled dataset:")
    print(df_filled.head())


if __name__ == "__main__":
    main()
