# Output Directory

This directory contains all generated files from the training and prediction processes.

## Files Generated

### Training Output
- **`weights.csv`** - Trained model weights for 4 binary classifiers (One-vs-All)
  - Contains weights for: Gryffindor, Hufflepuff, Ravenclaw, Slytherin
  - Rows: Features (Bias + 10 courses)
  - Columns: Houses

- **`normalization_params.csv`** - Normalization parameters from training
  - Mean and standard deviation for each feature
  - Used to normalize test data consistently with training data

### Prediction Output
- **`houses.csv`** - Final predictions for the test dataset
  - Format: `Index, Hogwarts House`
  - Contains predicted house for each student in test set

## File Dependencies

```
logreg_train.py  →  weights.csv
                 →  normalization_params.csv

logreg_predict.py  (uses)  →  weights.csv
                            →  normalization_params.csv
                   (creates) →  houses.csv
```

## Notes

- All files in this directory are auto-generated
- Do not manually edit these files
- Files are regenerated each time training/prediction is run
- `normalization_params.csv` is critical for prediction consistency
