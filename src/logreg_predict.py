#!/usr/bin/env python3
"""
Predict Hogwarts Houses using trained logistic regression model.
Uses One-vs-All approach with 4 binary classifiers.
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'utils')
from preprocess import load_and_preprocess_test_data


def load_weights(filepath='output/weights.csv'):
    """
    Load trained weights from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to weights CSV file
        
    Returns:
    --------
    thetas : np.ndarray
        Weight matrix, shape (n_features, n_houses)
    house_names : list
        List of house names in the same order as columns
    """
    weights_df = pd.read_csv(filepath, index_col=0)
    house_names = weights_df.columns.tolist()
    thetas = weights_df.values  # Shape: (n_features, n_houses)
    
    print("=" * 80)
    print("WEIGHTS LOADED")
    print("=" * 80)
    print(f"Weights shape: {thetas.shape}")
    print(f"Houses: {house_names}")
    print(f"✓ Weights loaded successfully")
    
    return thetas, house_names


def predict(X_test, thetas, house_names):
    """
    Predict Hogwarts Houses using One-vs-All logistic regression.
    
    Parameters:
    -----------
    X_test : np.ndarray
        Test feature matrix with bias, shape (n_samples, n_features)
    thetas : np.ndarray
        Weight matrix, shape (n_features, n_houses)
    house_names : list
        List of house names
        
    Returns:
    --------
    predicted_houses : list
        List of predicted house names
    probabilities : np.ndarray
        Probability matrix, shape (n_samples, n_houses)
    """
    # 3. CALCUL DE PROBABILITÉ (Tout le monde d'un coup)
    # On transpose theta pour aligner les dimensions
    # X_test: (M, 11) × thetas.T: (11, 4) = z: (M, 4)
    z = np.dot(X_test, thetas)
    
    # Appliquer la sigmoïde
    probabilities = 1 / (1 + np.exp(-z))  # Shape (M, 4)
    
    # 4. CHOISIR LA MEILLEURE MAISON
    # Pour chaque étudiant, prendre l'index de la probabilité maximale
    best_house_indices = np.argmax(probabilities, axis=1)  # Renvoie [0, 2, 1, 0...]
    
    # 5. CONVERTIR EN NOMS
    predicted_houses = [house_names[i] for i in best_house_indices]
    
    print("\n" + "=" * 80)
    print("PREDICTIONS MADE")
    print("=" * 80)
    print(f"Total predictions: {len(predicted_houses)}")
    print(f"\nHouse distribution:")
    unique, counts = np.unique(predicted_houses, return_counts=True)
    for house, count in zip(unique, counts):
        print(f"  {house}: {count}")
    
    return predicted_houses, probabilities


def save_predictions(predictions, original_df, filepath='output/houses.csv'):
    """
    Save predictions to CSV file.
    
    Parameters:
    -----------
    predictions : list
        List of predicted house names
    original_df : pd.DataFrame
        Original test dataframe (to get Index column)
    filepath : str
        Output CSV file path
    """
    # Create output dataframe with Index and Hogwarts House
    output_df = pd.DataFrame({
        'Index': original_df['Index'].values,
        'Hogwarts House': predictions
    })
    
    # Save to CSV
    output_df.to_csv(filepath, index=False)
    
    print("\n" + "=" * 80)
    print("PREDICTIONS SAVED")
    print("=" * 80)
    print(f"✓ Predictions saved to: {filepath}")
    print(f"\nFirst 10 predictions:")
    print(output_df.head(10))
    
    return output_df


def main():
    """Main prediction pipeline."""
    print("=" * 80)
    print("HOGWARTS HOUSE PREDICTION")
    print("=" * 80)
    
    # 1. CHARGER LES DONNÉES ET LES POIDS
    X_test, df_test = load_and_preprocess_test_data(
        test_filepath='datasets/dataset_test.csv',
        params_filepath='output/normalization_params.csv'
    )
    
    thetas, house_names = load_weights(filepath='output/weights.csv')
    
    # 2. FAIRE LES PRÉDICTIONS
    predicted_houses, probabilities = predict(X_test, thetas, house_names)
    
    # 3. SAUVEGARDER LES RÉSULTATS
    output_df = save_predictions(predicted_houses, df_test, filepath='output/houses.csv')
    
    # Afficher quelques exemples avec probabilités
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS WITH PROBABILITIES")
    print("=" * 80)
    for i in range(min(10, len(predicted_houses))):
        print(f"\nStudent {i} (Index {df_test['Index'].iloc[i]}):")
        print(f"  Predicted: {predicted_houses[i]}")
        print(f"  Probabilities:")
        for j, house in enumerate(house_names):
            print(f"    {house}: {probabilities[i, j]:.4f}")
    
    print("\n" + "=" * 80)
    print("PREDICTION COMPLETE!")
    print("=" * 80)
    print(f"✓ {len(predicted_houses)} predictions made")
    print(f"✓ Results saved to 'output/houses.csv'")


if __name__ == "__main__":
    main()
