#!/usr/bin/env python3
"""
Setup Script for Logistic Regression Project
Prepares the environment and checks dependencies before running the project.
"""

import os
import sys
import subprocess


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
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} detected")
        print("⚠️  Python 3.8+ is recommended")
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
            print(f"❌ {display_name} is NOT installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print(f"\nTo install missing packages, run:")
        print(f"  pip install {' '.join(missing_packages)}")
        print(f"\nOr install from requirements:")
        print(f"  pip install -r requirement.txt")
        return False
    
    return True


def create_directories():
    """Create necessary directories if they don't exist."""
    print_step(3, "Creating necessary directories...")
    
    directories = [
        'output',
        'data_visualization/histograms',
        'data_visualization/pair_plot',
        'data_visualization/scatter_plots'
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
            print(f"❌ {description} NOT FOUND: {filepath}")
            all_present = False
    
    if not all_present:
        print("\n⚠️  Missing datasets! Please ensure dataset files are in the 'datasets/' directory.")
        return False
    
    return True


def check_scripts():
    """Check if main scripts exist."""
    print_step(5, "Checking main scripts...")
    
    scripts = [
        ('logreg_train.py', 'Training script'),
        ('logreg_predict.py', 'Prediction script'),
        ('utils/preprocess.py', 'Preprocessing utilities')
    ]
    
    all_present = True
    
    for filepath, description in scripts:
        if os.path.exists(filepath):
            print(f"✓ {description}: {filepath}")
        else:
            print(f"❌ {description} NOT FOUND: {filepath}")
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
                    print(f"  ⚠️  Could not remove {f}: {e}")
        else:
            print("  ✓ Keeping existing output files")
    else:
        print("  No existing output files found")


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
        print("\n✅ All checks passed!")
        print("✅ Project is ready to use!")
        print_next_steps()
        return 0
    else:
        print("\n⚠️  Some checks failed!")
        print("Please resolve the issues above before running the project.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed with error: {e}")
        sys.exit(1)
