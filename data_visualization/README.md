# Data Visualization Scripts

This directory contains all scripts for data exploration and visualization.

## Scripts

### `describe.py`
Statistical description of the dataset.
- Calculates mean, std, min, max, quartiles for each numerical feature
- Similar to `pandas.describe()` but implemented from scratch

**Usage:**
```bash
python3 data_visualization/describe.py
```

### `histogram.py`
Generates histograms for all numerical features.
- Shows distribution of grades for each course
- Helps identify which features may be useful for classification

**Usage:**
```bash
python3 data_visualization/histogram.py
```

**Output:** `data_visualization/histograms/histograms.png`

### `pair_plot.py`
Creates a pair plot (scatter plot matrix) of features.
- Shows relationships between different features
- Color-coded by house
- Useful for feature selection

**Usage:**
```bash
python3 data_visualization/pair_plot.py
```

**Output:** `data_visualization/pair_plot/pair_plot.png`

### `scatter_plot.py`
Generates scatter plots comparing two features.
- Customizable feature selection
- Shows how well features separate houses

**Usage:**
```bash
python3 data_visualization/scatter_plot.py
```

**Output:** `data_visualization/scatter_plots/*.png`

## Output Directories

- **`histograms/`** - Contains histogram images
- **`pair_plot/`** - Contains pair plot images
- **`scatter_plots/`** - Contains scatter plot images

## Purpose

These visualization scripts help with:
1. Data exploration
2. Feature selection
3. Understanding data distribution
4. Identifying correlations between features
5. Verifying which features best separate houses
