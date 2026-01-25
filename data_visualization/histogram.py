import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for WSL
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path so we can import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import (
    read_csv,
    detect_numeric_columns
)


def plot_histogram():
    # Create the directory if it doesn't exist
    if not os.path.exists("plot"):
        os.makedirs("plot")

    dataset = read_csv("datasets/dataset_train.csv")
    houses = ["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw"]
    house_colors = {
        "Gryffindor": "tab:red",
        "Slytherin": "tab:green",
        "Hufflepuff": "tab:orange",
        "Ravenclaw": "tab:blue",
    }

    skip_cols = [
        "Index", "Hogwarts House", "First Name",
        "Last Name", "Birthday", "Best Hand"
    ]
    features = detect_numeric_columns(dataset, skip_columns=skip_cols)

    # Compute the subplot grid
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    # Create a single figure with all subplots
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows)
    )
    if n_rows > 1 or n_cols > 1:
        axes_flat = axes.ravel()
    else:
        axes_flat = [axes]

    for idx, feature in enumerate(features):
        ax = axes_flat[idx]

        for house in houses:
            values = [
                float(row[feature])
                for row in dataset.rows
                if (row.get("Hogwarts House") == house and
                    row.get(feature, "") != "")
            ]
            if values:  # Only plot if we have data
                ax.hist(
                    values, alpha=0.7, label=house, bins=30,
                    color=house_colors.get(house)
                )

        ax.set_title(feature)
        ax.set_xlabel("Grades")
        ax.set_ylabel("Number of students")
        ax.legend(fontsize="small")

    # Hide unused axes
    for ax in axes_flat[n_features:]:
        ax.set_visible(False)

    fig.tight_layout()
    plt.savefig("plot/histograms.png", dpi=150, bbox_inches="tight")
    print("Histograms saved to: plot/histograms.png")
    plt.close()


if __name__ == "__main__":
    plot_histogram()
