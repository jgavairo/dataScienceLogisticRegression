# Logistic Regression - Hogwarts House Prediction

Ce projet implémente une régression logistique pour prédire la maison de Poudlard (Hogwarts) d'un étudiant en utilisant ses notes dans différentes matières.

## � Quick Start

### 1. Setup (First time only)
```bash
python3 setup.py
```
Ce script va :
- Vérifier la version de Python
- Vérifier les dépendances (pandas, numpy, matplotlib)
- Créer les dossiers nécessaires
- Vérifier la présence des datasets
- Préparer l'environnement

### 2. Train the model
```bash
python3 logreg_train.py
```

### 3. Make predictions
```bash
python3 logreg_predict.py
```

### 4. Check results
```bash
head output/houses.csv
```

## 📁 Structure du Projet

```
.
├── logreg_train.py          # Script principal d'entraînement
├── logreg_predict.py        # Script principal de prédiction
├── setup.py                 # Script de configuration initiale
├── README.md                # Ce fichier
│
├── datasets/                # Données brutes
│   ├── dataset_train.csv    # Dataset d'entraînement
│   └── dataset_test.csv     # Dataset de test
│
├── output/                  # Fichiers générés
│   ├── weights.csv          # Poids du modèle entraîné
│   ├── normalization_params.csv  # Paramètres de normalisation
│   └── houses.csv           # Prédictions finales
│
├── data_visualization/      # Scripts de visualisation
│   ├── describe.py
│   ├── histogram.py
│   ├── pair_plot.py
│   └── scatter_plot.py
│
└── utils/                   # Utilitaires
    ├── preprocess.py        # Fonctions de preprocessing
    ├── pipeline.py          # Pipeline automatique
    ├── prepare_test_data.py
    └── check_prerequisites.py
```

## 📖 Scripts Principaux

### `logreg_train.py`
**Entraîne le modèle de régression logistique**

Fonctionnalités :
- Charge `dataset_train.csv`
- Remplit automatiquement les valeurs manquantes (avec moyennes)
- Normalise les données (z-score: `(x - mean) / std`)
- Ajoute le biais (colonne de 1)
- Entraîne 4 classificateurs binaires (One-vs-All)
- Utilise la descente de gradient pour optimiser les poids
- Sauvegarde `output/weights.csv` et `output/normalization_params.csv`

**Utilisation :**
```bash
python3 logreg_train.py
# ou avec un chemin custom :
python3 logreg_train.py datasets/dataset_train.csv
```

### `logreg_predict.py`
**Fait des prédictions sur le dataset de test**

Fonctionnalités :
- Charge `dataset_test.csv`
- Préprocesse automatiquement (fill missing + normalize avec les paramètres du train)
- Ajoute le biais
- Utilise les poids entraînés pour prédire
- Applique One-vs-All : choisit la maison avec la probabilité maximale
- Sauvegarde `output/houses.csv`

**Utilisation :**
```bash
python3 logreg_predict.py
```

**Sortie :**
```csv
Index,Hogwarts House
0,Hufflepuff
1,Ravenclaw
2,Gryffindor
...
```

### `setup.py`
**Configure l'environnement du projet**

Vérifie :
- Version de Python (3.8+)
- Dépendances installées (pandas, numpy, matplotlib)
- Présence des datasets
- Présence des scripts
- Crée les dossiers nécessaires

**Utilisation :**
```bash
python3 setup.py
```

## 🧮 Algorithme

### Régression Logistique One-vs-All
Module utilitaire contenant les fonctions de preprocessing :
- `load_normalization_params()` - Charge les paramètres de normalisation
- `clean_and_normalize()` - Nettoie et normalise les données
- `add_bias()` - Ajoute la colonne de biais
- `load_and_preprocess_test_data()` - Pipeline complet pour les données de test

#### 6. **pipeline.py**
Script orchestrateur qui exécute tout le pipeline automatiquement.

**Utilisation :**
```bash
python3 pipeline.py
```

**Options :**
1. Pipeline complet (entraînement + prédiction)
2. Prédiction rapide (utilise le modèle existant)
3. Entraînement uniquement
4. Quitter

## 🚀 Guide d'Utilisation Rapide

### Option 1 : Pipeline Automatique
```bash
python3 pipeline.py
# Choisir l'option 1 pour tout exécuter
```

### Option 2 : Étape par Étape

#### Entraînement
```bash
# Étape 1 : Remplir les valeurs manquantes
python3 fill_missing_values.py

# Étape 2 : Normaliser les données
python3 normalize_data.py

# Étape 3 : Entraîner le modèle
python3 train_model.py
```

#### Prédiction
```bash
# Faire des prédictions sur le dataset de test
python3 predict.py
```

## 🧮 Algorithme de Régression Logistique

### Approche One-vs-All
Pour classifier 4 maisons, on entraîne 4 modèles binaires :
- Gryffindor vs All
- Hufflepuff vs All
- Ravenclaw vs All
- Slytherin vs All

### Descente de Gradient
Pour chaque modèle, on itère :

1. **Produit scalaire** : `Z = X · θ`
2. **Activation (Sigmoïde)** : `H = 1 / (1 + e^(-Z))`
3. **Calcul du gradient** : `Gradient = (1/m) * (X^T · (H - y))`
4. **Mise à jour** : `θ = θ - (learning_rate × Gradient)`

### Features Utilisées
Les 10 matières sélectionnées :
- Astronomy
- Herbology
- Divination
- Muggle Studies
- Ancient Runes
- History of Magic
- Transfiguration
- Potions
- Charms
- Flying

## 📊 Fichiers de Sortie

### weights.csv
Contient les poids optimisés pour chaque maison.
- Lignes : Features (Bias + 10 matières)
- Colonnes : Maisons (Gryffindor, Hufflepuff, Ravenclaw, Slytherin)

### houses.csv
Contient les prédictions finales :
- Colonne 1 : Index de l'étudiant
- Colonne 2 : Maison prédite

## 🔧 Paramètres Ajustables

Dans `train_model.py`, fonction `main()` :
- `learning_rate` : Taux d'apprentissage (défaut : 0.1)
- `max_iter` : Nombre d'itérations (défaut : 1000)

## 📝 Notes Importantes

1. **Ordre des opérations** : Toujours suivre l'ordre :
   - Fill missing values → Normalize → Train
   
2. **Consistency** : Les données de test doivent être préprocessées avec les **mêmes paramètres** que les données d'entraînement (mean et std sauvegardés).

3. **Biais** : La colonne de biais (1) est essentielle pour permettre au modèle d'avoir un terme d'ordonnée à l'origine.

## 🎯 Évaluation

Pour évaluer les performances, comparez `houses.csv` avec les vraies étiquettes (si disponibles) en calculant l'accuracy :

```python
accuracy = (nombre_de_prédictions_correctes) / (nombre_total_de_prédictions)
```

## 📚 Dépendances

- pandas
- numpy

Installation :
```bash
pip install pandas numpy
```

## 🎓 Auteur

Projet de Data Science - Régression Logistique
