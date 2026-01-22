#!/usr/bin/env python3
"""
Check Prerequisites - Verify all required files exist before prediction
"""

import os
import sys


def check_file(filepath, description):
    """Check if a file exists and print status."""
    if os.path.exists(filepath):
        print(f"✓ {description}")
        print(f"  Path: {filepath}")
        return True
    else:
        print(f"❌ {description}")
        print(f"  Path: {filepath}")
        print(f"  Status: NOT FOUND")
        return False


def check_prerequisites():
    """Check all prerequisites for running predict.py"""
    print("=" * 80)
    print("CHECKING PREREQUISITES FOR PREDICTION")
    print("=" * 80)
    
    all_ok = True
    
    print("\n1. TRAINING DATA FILES:")
    print("-" * 80)
    all_ok &= check_file(
        'datasets/dataset_train.csv',
        'Original training dataset'
    )
    all_ok &= check_file(
        'datasets/dataset_train_filled_normalized.csv',
        'Normalized training dataset'
    )
    all_ok &= check_file(
        'datasets/dataset_train_filled_normalized_params.csv',
        'Normalization parameters (REQUIRED for test preprocessing)'
    )
    
    print("\n2. TRAINED MODEL:")
    print("-" * 80)
    all_ok &= check_file(
        'weights.csv',
        'Trained model weights (REQUIRED for prediction)'
    )
    
    print("\n3. TEST DATA:")
    print("-" * 80)
    all_ok &= check_file(
        'datasets/dataset_test.csv',
        'Test dataset (REQUIRED)'
    )
    
    print("\n4. REQUIRED SCRIPTS:")
    print("-" * 80)
    all_ok &= check_file(
        'logreg_predict.py',
        'Prediction script (REQUIRED)'
    )
    all_ok &= check_file(
        'preprocess.py',
        'Preprocessing utilities'
    )
    
    print("\n" + "=" * 80)
    if all_ok:
        print("✓ ALL PREREQUISITES MET")
        print("=" * 80)
        print("\nYou can run:")
        print("  python3 logreg_predict.py")
        print("\nThis will create:")
        print("  houses.csv - containing the predictions")
        return True
    else:
        print("❌ MISSING PREREQUISITES")
        print("=" * 80)
        print("\nPlease run the training pipeline first:")
        print("  1. python3 fill_missing_values.py")
        print("  2. python3 normalize_data.py")
        print("  3. python3 train_model.py")
        print("\nOr use the automated pipeline:")
        print("  python3 pipeline.py")
        return False


def check_training_prerequisites():
    """Check prerequisites for training."""
    print("=" * 80)
    print("CHECKING PREREQUISITES FOR TRAINING")
    print("=" * 80)
    
    all_ok = True
    
    print("\n1. REQUIRED DATA:")
    print("-" * 80)
    all_ok &= check_file(
        'datasets/dataset_train.csv',
        'Training dataset'
    )
    
    print("\n2. REQUIRED SCRIPTS:")
    print("-" * 80)
    all_ok &= check_file(
        'fill_missing_values.py',
        'Fill missing values script'
    )
    all_ok &= check_file(
        'normalize_data.py',
        'Normalization script'
    )
    all_ok &= check_file(
        'train_model.py',
        'Training script'
    )
    
    print("\n" + "=" * 80)
    if all_ok:
        print("✓ READY FOR TRAINING")
        print("=" * 80)
        print("\nRun in this order:")
        print("  1. python3 fill_missing_values.py")
        print("  2. python3 normalize_data.py")
        print("  3. python3 train_model.py")
        return True
    else:
        print("❌ MISSING FILES")
        print("=" * 80)
        return False


def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == '--train':
        check_training_prerequisites()
    else:
        check_prerequisites()


if __name__ == "__main__":
    main()
