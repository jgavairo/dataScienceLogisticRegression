import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path so we can import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import read_csv


def plot_scatter():
    dataset = read_csv("datasets/dataset_train.csv")

    if not os.path.exists("plot"):
        os.makedirs("plot")

    feat_x = "Astronomy"
    feat_y = "Defense Against the Dark Arts"

    color_map = {
        "Gryffindor": "red",
        "Slytherin": "green",
        "Hufflepuff": "orange",
        "Ravenclaw": "blue"
    }

    plt.figure(figsize=(10, 7))
    for house in color_map.keys():
        house_x = []
        house_y = []
        for row in dataset.rows:
            if (row.get("Hogwarts House") == house and
                    row.get(feat_x) and row.get(feat_y)):
                house_x.append(float(row[feat_x]))
                house_y.append(float(row[feat_y]))
        if house_x and house_y:
            plt.scatter(
                house_x, house_y, c=color_map[house],
                alpha=0.5, label=house, s=30
            )
    plt.title(f"Scatter Plot : {feat_x} vs {feat_y}")
    plt.xlabel(feat_x)
    plt.ylabel(feat_y)
    plt.legend(loc='best')

    plt.savefig("plot/scatter_plot.png")
    print("The scatter plot has been saved in plot/scatter_plot.png")
    plt.close()


if __name__ == "__main__":
    plot_scatter()
