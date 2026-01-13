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
from typing import Dict, List

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
    # Row order as in the subject example
    row_labels: List[str] = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]

    # Sort columns for deterministic display
    columns = sorted(stats.keys())

    # Header
    header_cells = [""] + columns
    print(" | ".join(header_cells))
    print("-+-".join("-" * len(cell) for cell in header_cells))

    # Lignes
    for label in row_labels:
        row = [label]
        key = label.lower() if label not in {"25%", "50%", "75%"} else label
        for col in columns:
            value = stats[col].get(key, float("nan"))
            row.append(format_float(value))
        print(" | ".join(row))


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print(f"Usage: {argv[0]} <path_to_dataset.csv>", file=sys.stderr)
        return 1

    csv_path = argv[1]

    # Clearly non-numeric columns to ignore
    skip_cols = ["Index", "Hogwarts House", "First Name", "Last Name", "Birthday", "Best Hand"]

    dataset: Dataset = read_csv(csv_path)
    stats = describe_dataset_numeric(dataset, skip_columns=skip_cols)

    if not stats:
        print("No numeric column detected.", file=sys.stderr)
        return 1

    print_describe_table(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


