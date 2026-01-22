import pandas as pd
import numpy as np
import sys


def fill_missing_values(df, selected_features):
    """
    Fill missing values in selected features with column means.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    selected_features : list
        List of feature names
        
    Returns:
    --------
    df_filled : pd.DataFrame
        Dataframe with missing values filled
    means : dict
        Dictionary of mean values for each feature
    """
    df_filled = df.copy()
    means = {}
    
    print("\n" + "=" * 80)
    print("FILLING MISSING VALUES")
    print("=" * 80)
    
    for feature in selected_features:
        if feature in df_filled.columns:
            missing_count = df_filled[feature].isnull().sum()
            if missing_count > 0:
                mean_value = df_filled[feature].mean()
                df_filled[feature] = df_filled[feature].fillna(mean_value)
                means[feature] = mean_value
                print(f"✓ {feature}: Filled {missing_count} missing values with mean = {mean_value:.6f}")
            else:
                means[feature] = df_filled[feature].mean()
    
    return df_filled, means


def normalize_data(df, selected_features, means=None):
    """
    Normalize features using z-score normalization.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with filled values
    selected_features : list
        List of feature names
    means : dict, optional
        Pre-computed means (if None, will compute)
        
    Returns:
    --------
    df_normalized : pd.DataFrame
        Normalized dataframe
    params : dict
        Dictionary containing mean and std for each feature
    """
    df_normalized = df.copy()
    params = {}
    
    print("\n" + "=" * 80)
    print("NORMALIZING DATA (Z-SCORE)")
    print("=" * 80)
    
    for feature in selected_features:
        if feature in df_normalized.columns:
            mean_value = df_normalized[feature].mean()
            std_value = df_normalized[feature].std()
            
            params[feature] = {
                'mean': mean_value,
                'std': std_value
            }
            
            # Apply z-score normalization
            df_normalized[feature] = (df_normalized[feature] - mean_value) / std_value
            print(f"✓ {feature}: mean={mean_value:.6f}, std={std_value:.6f}")
    
    return df_normalized, params


def load_and_prepare_data(filepath='datasets/dataset_train.csv'):
    """
    Load the raw dataset, preprocess it, extract features, add bias unit, 
    and prepare One-vs-All targets.
    
    This function performs all preprocessing steps:
    1. Load raw data
    2. Fill missing values with means
    3. Normalize features (z-score)
    4. Add bias unit
    5. Prepare One-vs-All targets
    
    Parameters:
    -----------
    filepath : str
        Path to the raw training dataset CSV file
        
    Returns:
    --------
    X : np.ndarray
        Feature matrix with bias unit (shape: n_samples x (n_features + 1))
    y_dict : dict
        Dictionary containing One-vs-All encoded targets for each house
    houses : list
        List of unique Hogwarts Houses
    feature_names : list
        List of feature names (including 'Bias')
    normalization_params : dict
        Dictionary containing mean and std for each feature
    """
    print("=" * 80)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 80)
    print(f"Loading: {filepath}")
    
    # Load the raw dataset
    df = pd.read_csv(filepath)
    
    # Define the features to extract
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
    
    print(f"Original shape: {df.shape}")
    print(f"Selected features: {len(selected_features)}")
    
    # Step 1: Fill missing values
    df_filled, means = fill_missing_values(df, selected_features)
    
    # Step 2: Normalize data
    df_normalized, normalization_params = normalize_data(df_filled, selected_features, means)
    
    # Save normalization parameters for later use in prediction
    params_df = pd.DataFrame(normalization_params).T
    params_path = 'output/normalization_params.csv'
    params_df.to_csv(params_path)
    print(f"\n✓ Normalization parameters saved to: {params_path}")
    
    # Extract features (X) and target (y)
    X = df_normalized[selected_features].values  # Convert to numpy array
    y = df_normalized['Hogwarts House'].values
    
    print("\n" + "=" * 80)
    print("ADDING BIAS AND PREPARING TARGETS")
    print("=" * 80)
    
    # 1. AJOUT DU BIAIS (Bias Unit)
    # Ajouter une colonne de 1 au début de X
    # Cela permet au modèle d'avoir un terme d'ordonnée à l'origine (b dans ax + b)
    n_samples = X.shape[0]
    bias_column = np.ones((n_samples, 1))  # Colonne de 1
    X = np.hstack((bias_column, X))  # Ajouter la colonne de biais à gauche
    print(f"✓ Bias unit added (column of ones)")
    
    # Noms des features avec le biais
    feature_names = ['Bias'] + selected_features
    
    # 2. PRÉPARATION "ONE-VS-ALL" (Encodage des cibles)
    # Pour chaque maison, créer un vecteur binaire : 1 si c'est cette maison, 0 sinon
    houses = sorted(df['Hogwarts House'].unique())  # Liste des maisons uniques
    y_dict = {}
    
    print(f"✓ Houses found: {houses}")
    
    for house in houses:
        # Créer un vecteur binaire : 1 si la maison correspond, 0 sinon
        y_dict[house] = (y == house).astype(int)
        print(f"  {house}: {np.sum(y_dict[house])} positive samples")
    
    return X, y_dict, houses, feature_names, normalization_params


def sigmoid(z):
    """
    Fonction sigmoïde pour convertir les valeurs linéaires en probabilités.
    
    Parameters:
    -----------
    z : np.ndarray
        Valeurs linéaires (Z = X · θ)
        
    Returns:
    --------
    np.ndarray
        Probabilités entre 0 et 1
    """
    return 1 / (1 + np.exp(-z))


def train_logistic_regression(X, y, learning_rate=0.01, max_iter=1000, verbose=True):
    """
    Entraîne un modèle de régression logistique avec la descente de gradient.
    
    Parameters:
    -----------
    X : np.ndarray
        Matrice de features avec biais (shape: n_samples x n_features)
    y : np.ndarray
        Vecteur cible binaire (0 ou 1)
    learning_rate : float
        Taux d'apprentissage (alpha)
    max_iter : int
        Nombre d'itérations de la descente de gradient
    verbose : bool
        Afficher les informations pendant l'entraînement
        
    Returns:
    --------
    theta : np.ndarray
        Vecteur de poids optimisés
    """
    # Initialisation
    m, n = X.shape  # m = nombre d'échantillons, n = nombre de features (avec biais)
    theta = np.zeros(n)  # Initialiser theta à zéro
    
    if verbose:
        print(f"  Starting training: {m} samples, {n} features")
        print(f"  Learning rate: {learning_rate}, Max iterations: {max_iter}")
    
    # 3. LA BOUCLE D'ENTRAÎNEMENT (Gradient Descent)
    for iteration in range(max_iter):
        # A. LE PRODUIT SCALAIRE (Le modèle linéaire)
        # Z = X · θ
        Z = np.dot(X, theta)
        
        # B. LA PRÉDICTION (L'activation)
        # H = 1 / (1 + e^(-Z))
        H = sigmoid(Z)
        
        # C. LE CALCUL DU GRADIENT (La direction de l'erreur)
        # Gradient = (1/m) * (X^T · (H - y))
        gradient = np.dot(X.T, (H - y)) / m
        
        # D. LA MISE À JOUR (L'apprentissage)
        # θ_nouveau = θ_ancien - (learning_rate × Gradient)
        theta = theta - (learning_rate * gradient)
        
        # Affichage périodique (tous les 100 itérations)
        if verbose and (iteration % 100 == 0 or iteration == max_iter - 1):
            # Calculer la loss (coût) pour suivre la progression
            # Loss = -1/m * Σ(y*log(H) + (1-y)*log(1-H))
            epsilon = 1e-15  # Pour éviter log(0)
            H_clipped = np.clip(H, epsilon, 1 - epsilon)
            loss = -np.mean(y * np.log(H_clipped) + (1 - y) * np.log(1 - H_clipped))
            print(f"    Iteration {iteration:4d}: Loss = {loss:.6f}")
    
    return theta


def train_all_houses(X, y_dict, houses, learning_rate=0.01, max_iter=1000):
    """
    Entraîne un modèle de régression logistique pour chaque maison (One-vs-All).
    
    Parameters:
    -----------
    X : np.ndarray
        Matrice de features avec biais
    y_dict : dict
        Dictionnaire contenant les cibles One-vs-All pour chaque maison
    houses : list
        Liste des maisons
    learning_rate : float
        Taux d'apprentissage
    max_iter : int
        Nombre d'itérations
        
    Returns:
    --------
    theta_dict : dict
        Dictionnaire contenant les poids optimisés pour chaque maison
    """
    theta_dict = {}
    
    print("\n" + "=" * 80)
    print("TRAINING LOGISTIC REGRESSION MODELS (ONE-VS-ALL)")
    print("=" * 80)
    
    for house in houses:
        print(f"\n{'=' * 80}")
        print(f"Training model for: {house} vs All")
        print(f"{'=' * 80}")
        
        # Récupérer les cibles pour cette maison
        y = y_dict[house]
        
        # Entraîner le modèle
        theta = train_logistic_regression(X, y, learning_rate, max_iter, verbose=True)
        
        # Stocker les poids
        theta_dict[house] = theta
        
        print(f"✓ Training completed for {house}")
        print(f"  Theta shape: {theta.shape}")
        print(f"  Theta (first 5 values): {theta[:5]}")
    
    return theta_dict


def save_weights(theta_dict, houses, feature_names, filepath='output/weights.csv'):
    """
    Sauvegarde les poids optimisés dans un fichier CSV.
    
    Parameters:
    -----------
    theta_dict : dict
        Dictionnaire contenant les poids pour chaque maison
    houses : list
        Liste des maisons
    feature_names : list
        Liste des noms de features
    filepath : str
        Chemin du fichier CSV de sortie
    """
    # Créer un DataFrame avec les poids
    # Chaque ligne = une feature, chaque colonne = une maison
    weights_df = pd.DataFrame(theta_dict, index=feature_names)
    
    # Sauvegarder
    weights_df.to_csv(filepath)
    
    print("\n" + "=" * 80)
    print("WEIGHTS SAVED")
    print("=" * 80)
    print(f"✓ Weights saved to: {filepath}")
    print(f"\nWeights DataFrame:")
    print(weights_df)
    
    return weights_df


def main():
    """Main function to load and prepare data for training."""
    # Parse command line arguments
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = 'datasets/dataset_train.csv'
    
    print("=" * 80)
    print("LOGISTIC REGRESSION TRAINING - ONE-VS-ALL")
    print("=" * 80)
    
    # Load and prepare data (includes preprocessing)
    X, y_dict, houses, feature_names, normalization_params = load_and_prepare_data(filepath)
    
    print("=" * 80)
    print("DATASET PREPARATION FOR LOGISTIC REGRESSION")
    print("=" * 80)
    
    # Informations sur X (matrice de features avec biais)
    print("\n" + "=" * 80)
    print("FEATURE MATRIX (X) WITH BIAS UNIT")
    print("=" * 80)
    print(f"Shape: {X.shape}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features (including bias): {X.shape[1]}")
    print(f"\nFeature names: {feature_names}")
    print(f"\nFirst 5 rows of X:")
    print(X[:5])
    
    # Vérification du biais
    print("\n" + "=" * 80)
    print("BIAS UNIT VERIFICATION")
    print("=" * 80)
    print(f"Bias column (first column):")
    print(f"  Min: {X[:, 0].min()}")
    print(f"  Max: {X[:, 0].max()}")
    print(f"  All ones? {np.all(X[:, 0] == 1)}")
    
    # Informations sur les cibles One-vs-All
    print("\n" + "=" * 80)
    print("ONE-VS-ALL TARGET ENCODING")
    print("=" * 80)
    print(f"Hogwarts Houses: {houses}")
    print(f"Number of binary classifiers needed: {len(houses)}")
    
    for house in houses:
        y_house = y_dict[house]
        print(f"\n{house}:")
        print(f"  Shape: {y_house.shape}")
        print(f"  Positive samples (1): {np.sum(y_house)}")
        print(f"  Negative samples (0): {len(y_house) - np.sum(y_house)}")
        print(f"  First 10 values: {y_house[:10]}")
    
    # Statistiques des features
    print("\n" + "=" * 80)
    print("FEATURE STATISTICS (excluding bias)")
    print("=" * 80)
    X_no_bias = X[:, 1:]  # Exclure la colonne de biais
    for i, feature_name in enumerate(feature_names[1:]):
        print(f"{feature_name}:")
        print(f"  Mean: {X_no_bias[:, i].mean():.6f}")
        print(f"  Std:  {X_no_bias[:, i].std():.6f}")
        print(f"  Min:  {X_no_bias[:, i].min():.6f}")
        print(f"  Max:  {X_no_bias[:, i].max():.6f}")
    
    print("\n" + "=" * 80)
    print("DATA READY FOR TRAINING")
    print("=" * 80)
    print("✓ Bias unit added (first column of X)")
    print("✓ One-vs-All targets created for each house")
    print("✓ Features are normalized (mean ≈ 0, std ≈ 1)")
    print("\nYou can now train 4 binary classifiers:")
    for house in houses:
        print(f"  - {house} vs All")
    
    # ENTRAÎNEMENT DES MODÈLES
    theta_dict = train_all_houses(
        X, 
        y_dict, 
        houses, 
        learning_rate=0.1,  # Taux d'apprentissage
        max_iter=1000        # Nombre d'itérations
    )
    
    # SAUVEGARDE DES POIDS
    weights_df = save_weights(theta_dict, houses, feature_names, filepath='output/weights.csv')
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"✓ Trained {len(houses)} binary classifiers")
    print(f"✓ Weights saved to 'output/weights.csv'")
    print("\nNext steps:")
    print("  1. Use these weights to make predictions on new data")
    print("  2. Evaluate model performance on test set")
    
    return X, y_dict, houses, feature_names, theta_dict


if __name__ == "__main__":
    main()
