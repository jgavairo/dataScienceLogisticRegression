import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path so we can import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import read_csv, detect_numeric_columns


def pair_plot():
    if not os.path.exists("plot"):
        os.makedirs("plot")

    dataset = read_csv("datasets/dataset_train.csv")
    
    houses = ["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw"]
    color_map = {
        "Gryffindor": "#ae0001", # Red
        "Slytherin": "#2a623d",  # Green
        "Hufflepuff": "#ffdb00", # Yellow
        "Ravenclaw": "#222f5b"   # Blue
    }

    skip_cols = ["Index", "Hogwarts House", "First Name", "Last Name", "Birthday", "Best Hand"]
    features = detect_numeric_columns(dataset, skip_columns=skip_cols)
    n = len(features)

    fig, axes = plt.subplots(n, n, figsize=(30, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i, feat_y in enumerate(features):
        for j, feat_x in enumerate(features):
            ax = axes[i, j]
            
            for house in houses:
                color = color_map[house]
                x_vals = []
                y_vals = []
                for row in dataset.rows:
                    if row["Hogwarts House"] == house and row[feat_x] != "" and row[feat_y] != "":
                        x_vals.append(float(row[feat_x]))
                        y_vals.append(float(row[feat_y]))

                if i == j:
                    ax.hist(x_vals, color=color, alpha=0.5, bins=20)
                else:
                    ax.scatter(x_vals, y_vals, color=color, s=0.5, alpha=0.4)
            
            if i == n - 1:
                ax.set_xlabel(feat_x, fontsize=10, rotation=45)
            if j == 0:
                ax.set_ylabel(feat_y, fontsize=10, rotation=45)
            
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Pair Plot: All Hogwarts Subjects Comparison", fontsize=30)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map[house], alpha=0.5, label=house)
        for house in houses
    ]
    fig.legend(
        handles=legend_elements, loc='upper right',
        bbox_to_anchor=(0.98, 0.98), fontsize=12
    )
    
    output_file = "plot/pair_plot.png"
    plt.savefig(output_file, dpi=150)
    print(f"Pair plot successfully saved as {output_file}")
    plt.close()


if __name__ == "__main__":
    pair_plot()