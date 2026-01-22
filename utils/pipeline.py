#!/usr/bin/env python3
"""
Pipeline Script - Complete Workflow for Logistic Regression
Runs the entire pipeline from data preparation to prediction.
"""

import os
import sys


def print_step(step_number, title):
    """Print a formatted step header."""
    print("\n" + "=" * 80)
    print(f"STEP {step_number}: {title}")
    print("=" * 80 + "\n")


def run_pipeline():
    """Run the complete machine learning pipeline."""
    print("=" * 80)
    print("LOGISTIC REGRESSION PIPELINE - HOGWARTS HOUSE PREDICTION")
    print("=" * 80)
    print("This pipeline will:")
    print("  1. Extract selected features from training data")
    print("  2. Fill missing values")
    print("  3. Normalize the data")
    print("  4. Train logistic regression models (One-vs-All)")
    print("  5. Prepare test data")
    print("  6. Make predictions")
    print("=" * 80)
    
    input("\nPress Enter to start the pipeline...")
    
    # Step 1: Extract features
    print_step(1, "EXTRACT FEATURES FROM TRAINING DATA")
    print("Running: train_model.py (extraction only)")
    print("This will create: datasets/dataset_train_selected.csv")
    # Note: The current train_model.py doesn't save selected data anymore
    # We'll skip this step as it's integrated in the next steps
    
    # Step 2: Fill missing values
    print_step(2, "FILL MISSING VALUES")
    print("Running: fill_missing_values.py")
    print("This will create: datasets/dataset_train_filled.csv")
    os.system("python3 fill_missing_values.py")
    
    input("\nStep 2 complete. Press Enter to continue...")
    
    # Step 3: Normalize data
    print_step(3, "NORMALIZE TRAINING DATA")
    print("Running: normalize_data.py")
    print("This will create:")
    print("  - datasets/dataset_train_filled_normalized.csv")
    print("  - datasets/dataset_train_filled_normalized_params.csv")
    os.system("python3 normalize_data.py")
    
    input("\nStep 3 complete. Press Enter to continue...")
    
    # Step 4: Train models
    print_step(4, "TRAIN LOGISTIC REGRESSION MODELS")
    print("Running: train_model.py")
    print("This will create: weights.csv")
    os.system("python3 train_model.py")
    
    input("\nStep 4 complete. Press Enter to continue...")
    
    # Step 5: Prepare test data
    print_step(5, "PREPARE TEST DATA")
    print("Running: prepare_test_data.py")
    print("This will create: datasets/dataset_test_prepared.csv")
    os.system("python3 prepare_test_data.py")
    
    input("\nStep 5 complete. Press Enter to continue...")
    
    # Step 6: Make predictions
    print_step(6, "MAKE PREDICTIONS")
    print("Running: logreg_predict.py")
    print("This will create: houses.csv")
    os.system("python3 logreg_predict.py")
    
    # Final summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  Training pipeline:")
    print("    - datasets/dataset_train_filled.csv")
    print("    - datasets/dataset_train_filled_normalized.csv")
    print("    - datasets/dataset_train_filled_normalized_params.csv")
    print("    - weights.csv")
    print("\n  Prediction pipeline:")
    print("    - datasets/dataset_test_prepared.csv")
    print("    - houses.csv (FINAL PREDICTIONS)")
    print("\n" + "=" * 80)


def quick_predict():
    """Quick prediction mode - assumes training is already done."""
    print("=" * 80)
    print("QUICK PREDICTION MODE")
    print("=" * 80)
    print("This assumes you already have:")
    print("  - weights.csv (trained model)")
    print("  - datasets/dataset_train_filled_normalized_params.csv (normalization params)")
    print("=" * 80)
    
    # Check if required files exist
    if not os.path.exists('weights.csv'):
        print("\n❌ ERROR: weights.csv not found!")
        print("Please run the training pipeline first.")
        return
    
    if not os.path.exists('datasets/dataset_train_filled_normalized_params.csv'):
        print("\n❌ ERROR: normalization parameters not found!")
        print("Please run the training pipeline first.")
        return
    
    print("\n✓ Required files found. Starting prediction...")
    
    # Run prediction directly
    os.system("python3 logreg_predict.py")
    
    print("\n" + "=" * 80)
    print("PREDICTION COMPLETE!")
    print("=" * 80)
    print("✓ Predictions saved to: houses.csv")


def main():
    """Main menu."""
    print("\n" + "=" * 80)
    print("HOGWARTS HOUSE PREDICTION - PIPELINE MANAGER")
    print("=" * 80)
    print("\nChoose an option:")
    print("  1. Run full pipeline (training + prediction)")
    print("  2. Quick prediction (use existing model)")
    print("  3. Train only (prepare data + train model)")
    print("  4. Exit")
    print("=" * 80)
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        run_pipeline()
    elif choice == "2":
        quick_predict()
    elif choice == "3":
        print("\nRunning training pipeline only...")
        os.system("python3 fill_missing_values.py")
        os.system("python3 normalize_data.py")
        os.system("python3 train_model.py")
        print("\n✓ Training complete! Run option 2 to make predictions.")
    elif choice == "4":
        print("\nGoodbye!")
        sys.exit(0)
    else:
        print("\n❌ Invalid choice. Please try again.")
        main()


if __name__ == "__main__":
    main()
