"""
Basic tools for the DSLR project.

Here we voluntarily limit ourselves to the standard library to respect
the subject constraints (no ready-made statistical functions).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass
class Dataset:
    """
    Simple representation of a tabular dataset.

    - header : column names in order.
    - rows   : list of rows, each row being a dict {col: value_str}.
    """

    header: List[str]
    rows: List[Dict[str, str]]


def read_csv(path: str) -> Dataset:
    """
    Reads a CSV file with a header and returns a Dataset.
    We don't convert types yet here to keep parsing generic.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows: List[Dict[str, str]] = [dict(row) for row in reader]
    return Dataset(header=header, rows=rows)


def is_float(value: str) -> bool:
    """Returns True if the string can be converted to float (excluding empty values)."""
    if value is None:
        return False
    value = value.strip()
    if value == "":
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False


def detect_numeric_columns(dataset: Dataset, skip_columns: Iterable[str] | None = None) -> List[str]:
    """
    Detects numeric columns by trying to parse values as float.

    - skip_columns : columns explicitly ignored (Index, names, etc.).
    """
    skip = set(skip_columns or [])
    numeric_cols: List[str] = []

    for col in dataset.header:
        if col in skip:
            continue
        # We consider a column numeric if all non-empty values
        # encountered in the rows are convertible to float.
        all_numeric = True
        for row in dataset.rows:
            v = row.get(col, "")
            if v is None or v == "":
                continue
            if not is_float(v):
                all_numeric = False
                break
        if all_numeric:
            numeric_cols.append(col)
    return numeric_cols


def column_values_as_floats(dataset: Dataset, column: str) -> List[float]:
    """
    Returns all non-empty values of a column converted to float.
    Empty or non-convertible values are ignored.
    """
    values: List[float] = []
    for row in dataset.rows:
        raw = row.get(column, "")
        if not is_float(raw):
            continue
        values.append(float(raw))  # type: ignore[arg-type]
    return values


def count(values: List[float]) -> int:
    """Number of observations."""
    return len(values)


def mean(values: List[float]) -> float:
    """Arithmetic mean."""
    n = len(values)
    if n == 0:
        return math.nan
    s = 0.0
    for v in values:
        s += v
    return s / n


def variance(values: List[float], sample: bool = True) -> float:
    """
    Variance.
    - sample=True -> sample variance (divide by n-1).
    - sample=False -> population variance (divide by n).
    """
    n = len(values)
    if n == 0:
        return math.nan
    if sample and n < 2:
        return math.nan
    m = mean(values)
    acc = 0.0
    for v in values:
        diff = v - m
        acc += diff * diff
    if sample:
        return acc / (n - 1)
    return acc / n


def std(values: List[float], sample: bool = True) -> float:
    """Standard deviation as square root of variance."""
    var = variance(values, sample=sample)
    if math.isnan(var):
        return math.nan
    return math.sqrt(var)


def min_max(values: List[float]) -> Tuple[float, float]:
    """Returns (min, max)."""
    if not values:
        return math.nan, math.nan
    vmin = values[0]
    vmax = values[0]
    for v in values[1:]:
        if v < vmin:
            vmin = v
        if v > vmax:
            vmax = v
    return vmin, vmax


def percentile(values: List[float], p: float) -> float:
    """
    Computes a percentile p (between 0 and 100) by linear interpolation.
    - We sort first.
    - We use indexing on (n - 1) to keep it simple.
    """
    if not values:
        return math.nan
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    # Theoretical position in the sorted vector
    pos = (p / 100.0) * (n - 1)
    lower_index = int(math.floor(pos))
    upper_index = int(math.ceil(pos))

    if lower_index == upper_index:
        return sorted_vals[lower_index]

    lower_value = sorted_vals[lower_index]
    upper_value = sorted_vals[upper_index]
    weight = pos - lower_index
    return lower_value + (upper_value - lower_value) * weight


def describe_numeric_column(values: List[float]) -> Dict[str, float]:
    """
    Computes all requested statistics for a numeric column.
    Returns a dict with keys:
    - count, mean, std, min, 25%, 50%, 75%, max
    """
    if not values:
        # Return NaN everywhere, count=0
        return {
            "count": 0.0,
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "25%": math.nan,
            "50%": math.nan,
            "75%": math.nan,
            "max": math.nan,
        }

    c = float(count(values))
    m = mean(values)
    s = std(values, sample=True)
    vmin, vmax = min_max(values)
    q25 = percentile(values, 25.0)
    q50 = percentile(values, 50.0)
    q75 = percentile(values, 75.0)

    return {
        "count": c,
        "mean": m,
        "std": s,
        "min": vmin,
        "25%": q25,
        "50%": q50,
        "75%": q75,
        "max": vmax,
    }


def describe_dataset_numeric(dataset: Dataset, skip_columns: Iterable[str] | None = None) -> Dict[str, Dict[str, float]]:
    """
    Computes describe statistics for all numeric columns
    and returns a mapping {column_name: stats_dict}.
    """
    numeric_cols = detect_numeric_columns(dataset, skip_columns=skip_columns)
    result: Dict[str, Dict[str, float]] = {}
    for col in numeric_cols:
        values = column_values_as_floats(dataset, col)
        result[col] = describe_numeric_column(values)
    return result



