# Datasets

This directory contains the raw datasets used for training and prediction.

## Files

### `dataset_train.csv`
**Raw training dataset** - Used by `logreg_train.py`

- **Size:** 1600 students
- **Features:** 19 columns including:
  - Student information: Index, Name, Birthday, Best Hand
  - Target: Hogwarts House (Gryffindor, Hufflepuff, Ravenclaw, Slytherin)
  - 13 course grades: Arithmancy, Astronomy, Herbology, Defense Against the Dark Arts, Divination, Muggle Studies, Ancient Runes, History of Magic, Transfiguration, Potions, Care of Magical Creatures, Charms, Flying

- **Missing values:** Yes, handled automatically by `logreg_train.py`

### `dataset_test.csv`
**Raw test dataset** - Used by `logreg_predict.py`

- **Size:** 400 students
- **Features:** Same as training dataset
- **Target:** Hogwarts House column is empty (to be predicted)
- **Missing values:** Yes, handled automatically by `logreg_predict.py`

## Important Notes

### ⚠️ Do Not Modify
These are the **original raw datasets**. Do not modify or preprocess them manually.

### 🔄 Preprocessing is Automatic
All preprocessing (filling missing values, normalization) is done automatically by:
- **Training:** `logreg_train.py` preprocesses `dataset_train.csv`
- **Prediction:** `logreg_predict.py` preprocesses `dataset_test.csv`

### 📊 No Intermediate Files
There are no intermediate preprocessed files in this directory. All preprocessing happens in memory during execution.

## Workflow

```
dataset_train.csv  →  logreg_train.py  →  output/weights.csv
                                        →  output/normalization_params.csv

dataset_test.csv   →  logreg_predict.py  →  output/houses.csv
                      (uses output/weights.csv)
                      (uses output/normalization_params.csv)
```

## Data Source

These datasets are part of the Hogwarts student records and contain grades from various magical courses. The goal is to predict which Hogwarts House a student belongs to based on their academic performance.
