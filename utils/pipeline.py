#!/usr/bin/env python3
"""
Pipeline Script - Complete Workflow for Logistic Regression
Runs the entire pipeline from training to evaluation.
"""

import os
import sys
import subprocess

try:
    import inquirer
    HAS_INQUIRER = True
except ImportError:
    HAS_INQUIRER = False


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)


def run_command(script_name):
    """Run a Python script and handle errors."""
    try:
        subprocess.run([sys.executable, script_name], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}: {e}")
        return False


def full_pipeline():
    """Run the complete machine learning pipeline (train + predict + evaluate)."""
    print_header("RUNNING COMPLETE PIPELINE")
    
    print("\n📋 Pipeline steps:")
    print("  1. Train the model")
    print("  2. Make predictions")
    print("  3. Evaluate accuracy")
    
    # Step 1: Train
    print("\n" + "=" * 80)
    print("STEP 1: TRAINING THE MODEL")
    print("=" * 80)
    print("▶️  Running: src/logreg_train.py")
    if not run_command("src/logreg_train.py"):
        return
    
    # Step 2: Predict
    print("\n" + "=" * 80)
    print("STEP 2: MAKING PREDICTIONS")
    print("=" * 80)
    print("▶️  Running: src/logreg_predict.py")
    if not run_command("src/logreg_predict.py"):
        return
    
    # Step 3: Evaluate
    print("\n" + "=" * 80)
    print("STEP 3: EVALUATING MODEL")
    print("=" * 80)
    print("▶️  Running: src/logreg_evaluate.py")
    if not run_command("src/logreg_evaluate.py"):
        return
    
    print("\n" + "=" * 80)
    print("✅ COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print("=" * 80)


def train_only():
    """Train the model only."""
    print_header("TRAINING THE MODEL")
    
    print("▶️  Running: src/logreg_train.py")
    run_command("src/logreg_train.py")
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("  - Run predictions: python3 src/logreg_predict.py")
    print("  - Evaluate model: python3 src/logreg_evaluate.py")


def predict_only():
    """Make predictions using existing model."""
    print_header("MAKING PREDICTIONS")
    
    if not os.path.exists('output/weights.csv'):
        print("❌ Model weights not found!")
        print("⚠️  Please train the model first")
        return
    
    print("▶️  Running: src/logreg_predict.py")
    run_command("src/logreg_predict.py")
    
    print("\n" + "=" * 80)
    print("✅ PREDICTIONS COMPLETE!")
    print("=" * 80)


def evaluate_only():
    """Evaluate the model."""
    print_header("EVALUATING THE MODEL")
    
    if not os.path.exists('output/houses.csv'):
        print("❌ Predictions not found!")
        print("⚠️  Please make predictions first")
        return
    
    if not os.path.exists('datasets/dataset_truth.csv'):
        print("❌ Ground truth dataset not found!")
        return
    
    print("▶️  Running: src/logreg_evaluate.py")
    run_command("src/logreg_evaluate.py")
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 80)


def interactive_menu():
    """Display interactive menu with arrow key navigation."""
    if not HAS_INQUIRER:
        print("\n⚠️  inquirer package not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "inquirer", "-q"])
            import importlib
            globals()['inquirer'] = importlib.import_module('inquirer')
        except Exception as e:
            print(f"❌ Could not install inquirer: {e}")
            return simple_menu()
    
    print_header("PIPELINE MANAGER - CHOOSE YOUR ACTION")
    
    questions = [
        inquirer.List(
            'action',
            message='Select an action (use arrow keys ↑↓ and press Enter)',
            choices=[
                ('🚀 Complete Pipeline (train + predict + evaluate)', 'full'),
                ('🎓 Train Only (train the model)', 'train'),
                ('🔮 Predict Only (use existing model)', 'predict'),
                ('📈 Evaluate Only (check accuracy)', 'evaluate'),
                ('❌ Exit', 'exit'),
            ],
            carousel=True,
        ),
    ]
    
    answers = inquirer.prompt(questions)
    
    if answers is None:
        return None
    
    return answers['action']


def simple_menu():
    """Simple text-based menu as fallback."""
    print_header("PIPELINE MANAGER - CHOOSE YOUR ACTION")
    print("""
1) 🚀 Complete Pipeline (train + predict + evaluate)
2) 🎓 Train Only (train the model)
3) 🔮 Predict Only (use existing model)
4) 📈 Evaluate Only (check accuracy)
5) ❌ Exit
    """)
    
    choice = input("Enter your choice (1-5): ").strip()
    
    mapping = {
        '1': 'full',
        '2': 'train',
        '3': 'predict',
        '4': 'evaluate',
        '5': 'exit'
    }
    
    return mapping.get(choice, None)


def main():
    """Main menu handler."""
    # Check if a direct action was passed as argument
    if len(sys.argv) > 1:
        action = sys.argv[1]
    else:
        try:
            action = interactive_menu()
            
            if action is None:
                print("\n⚠️  Menu cancelled")
                return
        except KeyboardInterrupt:
            print("\n\n⚠️  Pipeline interrupted by user")
            sys.exit(1)
    
    if action == 'full':
        full_pipeline()
    elif action == 'train':
        train_only()
    elif action == 'predict':
        predict_only()
    elif action == 'evaluate':
        evaluate_only()
    elif action == 'exit':
        print("\n👋 Goodbye!")
        return
    else:
        print("\n❌ Invalid action")
        return
    
    # Ask if user wants to continue (only if no argument was passed)
    if len(sys.argv) <= 1:
        print("\n" + "=" * 80)
        response = input("Would you like to run another action? [y/N]: ").strip().lower()
        if response == 'y':
            main()


if __name__ == "__main__":
    main()

