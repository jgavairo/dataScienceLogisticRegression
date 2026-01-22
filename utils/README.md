# Utility Scripts

This directory contains utility scripts that support the main training and prediction workflow.

## Scripts

### `preprocess.py`
Core preprocessing functions used by `logreg_predict.py`.

**Functions:**
- `load_normalization_params()` - Loads mean/std from training
- `clean_and_normalize()` - Fills missing values and normalizes data
- `add_bias()` - Adds bias column to feature matrix
- `load_and_preprocess_test_data()` - Complete test data preprocessing pipeline

**Usage:**
This is a module imported by `logreg_predict.py`, not run directly.

### `pipeline.py`
Automated workflow manager for the entire project.

**Features:**
- Interactive menu
- Full pipeline (training + prediction)
- Quick prediction mode
- Training-only mode

**Usage:**
```bash
python3 utils/pipeline.py
```

### `check_prerequisites.py`
Verifies that all required files exist before running training or prediction.

**Checks:**
- Dataset files
- Normalization parameters
- Trained weights
- Required scripts

**Usage:**
```bash
python3 utils/check_prerequisites.py          # Check for prediction
python3 utils/check_prerequisites.py --train  # Check for training
```

### `prepare_test_data.py`
Standalone script to preprocess test data (optional).

**Features:**
- Fills missing values with training means
- Normalizes using training parameters
- Saves prepared test data

**Usage:**
```bash
python3 utils/prepare_test_data.py
```

**Note:** This is optional as `logreg_predict.py` does preprocessing automatically.

## Purpose

These utilities:
1. Simplify the workflow
2. Ensure consistency between training and test preprocessing
3. Provide automation and verification tools
4. Make the project more user-friendly
