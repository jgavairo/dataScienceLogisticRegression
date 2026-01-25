#!/usr/bin/env python3
"""
Setup Script for Logistic Regression Project
Prepares the environment and checks dependencies before running the project.
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


def print_step(step_num, description):
    """Print a step description."""
    print(f"\n[{step_num}] {description}")


def check_python_version():
    """Check if Python version is adequate."""
    print_step(1, "Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"Python {version.major}.{version.minor}.{version.micro} detected")
        print("Python 3.8+ is recommended")
        return False
    
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def check_dependencies():
    """Check if required Python packages are installed."""
    print_step(2, "Checking dependencies...")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib (optional, for visualization)'
    }
    
    missing_packages = []
    
    for package, display_name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {display_name} is installed")
        except ImportError:
            print(f"{display_name} is NOT installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print(f"\nTo install missing packages, run:")
        print(f"  pip install {' '.join(missing_packages)}")
        print(f"\nOr install from requirements:")
        print(f"  pip install -r requirements/requirements.txt")
        return False
    
    return True


def create_directories():
    """Create necessary directories if they don't exist."""
    print_step(3, "Creating necessary directories...")
    
    directories = [
        'output',
        'plot'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"  Directory already exists: {directory}")
    
    return True


def check_datasets():
    """Check if required datasets exist."""
    print_step(4, "Checking datasets...")
    
    datasets = [
        ('datasets/dataset_train.csv', 'Training dataset'),
        ('datasets/dataset_test.csv', 'Test dataset')
    ]
    
    all_present = True
    
    for filepath, description in datasets:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✓ {description}: {filepath} ({size:,} bytes)")
        else:
            print(f"{description} NOT FOUND: {filepath}")
            all_present = False
    
    if not all_present:
        print("\nMissing datasets! Please ensure dataset files are in the 'datasets/' directory.")
        return False
    
    return True


def check_scripts():
    """Check if main scripts exist."""
    print_step(5, "Checking main scripts...")
    
    scripts = [
        ('src/logreg_train.py', 'Training script'),
        ('src/logreg_predict.py', 'Prediction script'),
        ('utils/preprocess.py', 'Preprocessing utilities')
    ]
    
    all_present = True
    
    for filepath, description in scripts:
        if os.path.exists(filepath):
            print(f"✓ {description}: {filepath}")
        else:
            print(f"{description} NOT FOUND: {filepath}")
            all_present = False
    
    return all_present


def clean_output_directory():
    """Clean old output files (optional)."""
    print_step(6, "Checking output directory...")
    
    output_files = [
        'output/weights.csv',
        'output/normalization_params.csv',
        'output/houses.csv'
    ]
    
    existing_files = [f for f in output_files if os.path.exists(f)]
    
    if existing_files:
        print(f"  Found {len(existing_files)} existing output file(s):")
        for f in existing_files:
            print(f"    - {f}")
        
        response = input("\n  Do you want to keep existing output files? [Y/n]: ").strip().lower()
        
        if response == 'n':
            for f in existing_files:
                try:
                    os.remove(f)
                    print(f"  ✓ Removed: {f}")
                except Exception as e:
                    print(f"  Could not remove {f}: {e}")
        else:
            print("  ✓ Keeping existing output files")
    else:
        print("  No existing output files found")


def cleanup_project():
    """Clean up all generated files and old project files."""
    print_header("CLEANUP PROJECT")
    
    files_to_clean = [
        # Generated output files
        ('output/weights.csv', 'Model weights'),
        ('output/normalization_params.csv', 'Normalization parameters'),
        ('output/houses.csv', 'Model predictions'),
        ('output/evaluation_report.txt', 'Evaluation report'),
        # Old root-level scripts (moved to src/)
        ('logreg_train.py', 'Old training script'),
        ('logreg_predict.py', 'Old prediction script'),
        ('logreg_evaluate.py', 'Old evaluation script'),
        # Plot files
        ('plot/histograms.png', 'Histogram plots'),
        ('plot/scatter_plot.png', 'Scatter plots'),
        ('plot/pair_plot.png', 'Pair plots'),
    ]
    
    # Find existing files
    existing_files = [(path, desc) for path, desc in files_to_clean if os.path.exists(path)]
    
    if not existing_files:
        print("\n✓ Project is already clean. No files to remove.")
        return
    
    print(f"\nFound {len(existing_files)} file(s) to clean:")
    for i, (path, desc) in enumerate(existing_files, 1):
        size = os.path.getsize(path) if os.path.isfile(path) else 0
        size_str = f"({size:,} bytes)" if size > 0 else ""
        print(f"  {i}. {desc}: {path} {size_str}")
    
    print("\nWarning: This will delete all generated files and old project files!")
    response = input("\nAre you sure you want to clean up? [y/N]: ").strip().lower()
    
    if response == 'y':
        removed_count = 0
        failed_count = 0
        
        for path, desc in existing_files:
            try:
                os.remove(path)
                print(f"✓ Removed: {path}")
                removed_count += 1
            except Exception as e:
                print(f"✗ Failed to remove {path}: {e}")
                failed_count += 1
        
        print(f"\n{removed_count} file(s) removed successfully")
        if failed_count > 0:
            print(f"{failed_count} file(s) failed to remove")
    else:
        print("\nCleanup cancelled.")


def print_next_steps():
    """Print instructions for next steps."""
    print_header("SETUP COMPLETE - NEXT STEPS")
    
    print("""
To train the model:
  $ python3 logreg_train.py
  
  This will:
    • Load and preprocess dataset_train.csv
    • Train 4 binary classifiers (One-vs-All)
    • Generate output/weights.csv
    • Generate output/normalization_params.csv

To make predictions:
  $ python3 logreg_predict.py
  
  This will:
    • Load and preprocess dataset_test.csv
    • Use trained weights to predict houses
    • Generate output/houses.csv

Optional - Data visualization:
  $ python3 data_visualization/describe.py
  $ python3 data_visualization/histogram.py
  $ python3 data_visualization/pair_plot.py
  $ python3 data_visualization/scatter_plot.py

Optional - Automated pipeline:
  $ python3 utils/pipeline.py
    """)


def interactive_menu():
    """Display interactive menu with arrow key navigation."""
    if not HAS_INQUIRER:
        print("\ninquirer package not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "inquirer", "-q"])
            import importlib
            globals()['inquirer'] = importlib.import_module('inquirer')
        except Exception as e:
            print(f"Could not install inquirer: {e}")
            print("Please run: pip install inquirer")
            return simple_menu()
    
    print_header("WHAT WOULD YOU LIKE TO DO?")
    
    questions = [
        inquirer.List(
            'action',
            message='Choose an action (use arrow keys ↑↓ and press Enter)',
            choices=[
                ('Data Visualization (describe, histogram, scatter, pair plot)', 'visualization'),
                ('Train the Model (logistic regression with gradient descent)', 'train'),
                ('Make Predictions (predict houses on test dataset)', 'predict'),
                ('Evaluate Model (check accuracy on test dataset)', 'evaluate'),
                ('Complete Pipeline (train + predict + evaluate)', 'pipeline'),
                ('Cleanup Project (remove generated files and old scripts)', 'cleanup'),
                ('Exit', 'exit'),
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
    print_header("WHAT WOULD YOU LIKE TO DO?")
    print("""
1) Data Visualization (describe, histogram, scatter, pair plot)
2) Train the Model (logistic regression with gradient descent)
3) Make Predictions (predict houses on test dataset)
4) Evaluate Model (check accuracy on test dataset)
5) Complete Pipeline (train + predict + evaluate)
6) Cleanup Project (remove generated files and old scripts)
7) Exit
    """)
    
    choice = input("Enter your choice (1-7): ").strip()
    
    mapping = {
        '1': 'visualization',
        '2': 'train',
        '3': 'predict',
        '4': 'evaluate',
        '5': 'pipeline',
        '6': 'cleanup',
        '7': 'exit'
    }
    
    return mapping.get(choice, None)


def run_visualization():
    """Run all visualization scripts."""
    print_header("RUNNING DATA VISUALIZATION")
    
    scripts = [
        ('data_visualization/describe.py', 'Statistical Description', ['datasets/dataset_train.csv']),
        ('data_visualization/histogram.py', 'Histogram Visualization', []),
        ('data_visualization/scatter_plot.py', 'Scatter Plot Visualization', []),
        ('data_visualization/pair_plot.py', 'Pair Plot Visualization', []),
    ]
    
    for script, description, args in scripts:
        if os.path.exists(script):
            print(f"\nRunning: {description}")
            try:
                cmd = [sys.executable, script] + args
                # Run from the project root directory so imports work correctly
                subprocess.run(cmd, check=True, cwd=os.getcwd())
                print(f"✓ {description} completed")
            except subprocess.CalledProcessError as e:
                print(f"{description} failed: {e}")
        else:
            print(f"Script not found: {script}")
    
    print("\nVisualization complete!")


def run_training():
    """Run the training script."""
    print_header("TRAINING THE MODEL")
    
    script = 'src/logreg_train.py'
    
    if os.path.exists(script):
        print(f"Running: {script}")
        try:
            subprocess.run([sys.executable, script], check=True)
            print(f"\n✓ Training completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Training failed: {e}")
    else:
        print(f"Script not found: {script}")


def run_prediction():
    """Run the prediction script."""
    print_header("MAKING PREDICTIONS")
    
    script = 'src/logreg_predict.py'
    
    if not os.path.exists('output/weights.csv'):
        print("Model weights not found!")
        print("Please train the model first: python3 logreg_train.py")
        return
    
    if os.path.exists(script):
        print(f"Running: {script}")
        try:
            subprocess.run([sys.executable, script], check=True)
            print(f"\n✓ Predictions completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Prediction failed: {e}")
    else:
        print(f"Script not found: {script}")


def run_pipeline():
    """Run the complete pipeline."""
    print_header("RUNNING COMPLETE PIPELINE")
    
    script = 'utils/pipeline.py'
    
    if os.path.exists(script):
        print(f"Running: {script}")
        try:
            # Pass 'full' argument to skip the menu and run directly
            subprocess.run([sys.executable, script, 'full'], check=True)
            print(f"\n✓ Pipeline completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Pipeline failed: {e}")
    else:
        print(f"Script not found: {script}")


def run_evaluation():
    """Run the evaluation script."""
    print_header("EVALUATING MODEL")
    
    script = 'src/logreg_evaluate.py'
    
    if not os.path.exists('output/houses.csv'):
        print("Predictions not found!")
        print("Please make predictions first: python3 logreg_predict.py")
        return
    
    if not os.path.exists('datasets/dataset_truth.csv'):
        print("Ground truth dataset not found!")
        print("Please ensure datasets/dataset_truth.csv exists")
        return
    
    if os.path.exists(script):
        print(f"Running: {script}")
        try:
            subprocess.run([sys.executable, script], check=True)
            print(f"\n✓ Evaluation completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Evaluation failed: {e}")
    else:
        print(f"Script not found: {script}")


def handle_menu_action(action):
    """Handle the selected menu action."""
    if action == 'visualization':
        run_visualization()
    elif action == 'train':
        run_training()
    elif action == 'predict':
        run_prediction()
    elif action == 'evaluate':
        run_evaluation()
    elif action == 'pipeline':
        run_pipeline()
    elif action == 'cleanup':
        cleanup_project()
    elif action == 'exit':
        print("\n👋 Goodbye!")
        return False
    else:
        print("Invalid choice. Please try again.")
    
    return True


def main():
    """Main setup function."""
    print_header("LOGISTIC REGRESSION PROJECT SETUP")
    
    print("""
This script will:
  1. Check Python version
  2. Check required dependencies
  3. Create necessary directories
  4. Verify datasets are present
  5. Verify main scripts are present
  6. Prepare output directory
    """)
    
    # Run all checks
    checks = [
        check_python_version(),
        check_dependencies(),
        create_directories(),
        check_datasets(),
        check_scripts()
    ]
    
    # Clean output directory
    clean_output_directory()
    
    # Summary
    print_header("SETUP SUMMARY")
    
    if all(checks):
        print("\nAll checks passed!")
        print("Project is ready to use!")
        
        # Ask if user wants to continue to menu
        response = input("\n\nWould you like to access the interactive menu? [Y/n]: ").strip().lower()
        
        if response != 'n':
            print("\n" + "=" * 80)
            continue_menu = True
            while continue_menu:
                try:
                    action = interactive_menu()
                    if action is None:
                        print("\nMenu cancelled by user")
                        break
                    continue_menu = handle_menu_action(action)
                except KeyboardInterrupt:
                    print("\n\nMenu interrupted by user")
                    break
        
        return 0
    else:
        print("\nSome checks failed!")
        print("Please resolve the issues above before running the project.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nSetup failed with error: {e}")
        sys.exit(1)
