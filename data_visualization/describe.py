#!/usr/bin/env python3
"""
Describe script for the DSLR project.

Usage:
    python describe.py datasets/dataset_train.csv

Displays, for each numeric feature, a statistics table
similar to pandas.DataFrame.describe() but implemented manually,
as required by the subject.
"""

from __future__ import annotations

import sys
import os
from typing import Dict, List

# Add parent directory to path so we can import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import Dataset, describe_dataset_numeric, read_csv


def format_float(value: float) -> str:
    """
    Formats a float with 6 decimals, or leaves empty if NaN.
    """
    if value != value:  # test NaN sans importer math
        return ""
    return f"{value:.6f}"


def print_describe_table(stats: Dict[str, Dict[str, float]]) -> None:
    """
    Displays a table like:

                Feature1    Feature2 ...
        Count   ...         ...
        Mean    ...         ...
        ...
    """
    # Ordre des lignes comme dans l'exemple du sujet
    row_labels: List[str] = [
        "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"
    ]

    # Colonnes triées pour un affichage déterministe
    columns = sorted(stats.keys())

    # Calcul des largeurs de colonnes pour un bel alignement
    # Largeur de la colonne des labels de lignes
    row_label_width = max(len(label) for label in row_labels)

    # Largeurs pour chaque colonne de feature
    # (en fonction du nom ET des valeurs formatées)
    col_widths: Dict[str, int] = {}
    for col in columns:
        max_width = len(col)
        for label in row_labels:
            if label not in {"25%", "50%", "75%"}:
                key = label.lower()
            else:
                key = label
            value = stats[col].get(key, float("nan"))
            cell = format_float(value)
            if len(cell) > max_width:
                max_width = len(cell)
        col_widths[col] = max_width

    # En-tête
    header = (" " * row_label_width + " | " +
              " | ".join(f"{col:>{col_widths[col]}}" for col in columns))
    separator = ("-" * row_label_width + "-+-" +
                 "-+-".join("-" * col_widths[col] for col in columns))
    print(header)
    print(separator)

    # Corps du tableau
    for label in row_labels:
        key = label.lower() if label not in {"25%", "50%", "75%"} else label
        cells = []
        for col in columns:
            value = stats[col].get(key, float("nan"))
            cells.append(f"{format_float(value):>{col_widths[col]}}")
        line = f"{label:<{row_label_width}} | " + " | ".join(cells)
        print(line)


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print(f"Usage: {argv[0]} <path_to_dataset.csv>", file=sys.stderr)
        return 1

    csv_path = argv[1]

    # Clearly non-numeric columns to ignore
    skip_cols = [
        "Index", "Hogwarts House", "First Name",
        "Last Name", "Birthday", "Best Hand"
    ]

    dataset: Dataset = read_csv(csv_path)
    stats = describe_dataset_numeric(dataset, skip_columns=skip_cols)

    if not stats:
        print("No numeric column detected.", file=sys.stderr)
        return 1

    print_describe_table(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

