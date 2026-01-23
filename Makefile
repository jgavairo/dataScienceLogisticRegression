# Makefile to run the visualization scripts
# Usage:
#   make               # runs all visualizations (run_all)
#   make run_all       # same as above
#   make describe      # run only describe
#   make histogram     # run only histogram
#   make scatter       # run only scatter
#   make pair          # run only pair_plot
#   make help          # show this help
# You can override the dataset path with DATASET=path/to/file.csv

DATASET ?= datasets/dataset_train.csv
TRAIN_DATA ?= datasets/dataset_train.csv
TEST_DATA ?= datasets/dataset_test.csv
PYTHON ?= python3

.PHONY: all help describe histogram scatter pair run_all clean train predict evaluate

all: run_all

describe:
	$(PYTHON) -m data_visualization.describe $(DATASET)

histogram:
	$(PYTHON) -m data_visualization.histogram $(DATASET)

scatter:
	$(PYTHON) -m data_visualization.scatter_plot $(DATASET)

pair:
	$(PYTHON) -m data_visualization.pair_plot $(DATASET)

visualization: describe histogram scatter pair

train:
	@echo "Running training with TRAIN_DATA=$(TRAIN_DATA)"
	$(PYTHON) logreg_train.py $(TRAIN_DATA)

predict:
	@echo "Running prediction (uses output/weights.csv and datasets/dataset_test.csv by default)"
	$(PYTHON) logreg_predict.py

evaluate:
	@echo "Running evaluation (compares output/houses.csv with datasets/dataset_truth.csv)"
	$(PYTHON) logreg_evaluate.py

help:
	@echo "Usage: make [target] [DATASET=path/to/file.csv]"
	@echo "Targets:"
	@echo "  describe  - run describe visualization"
	@echo "  histogram - run histogram visualization"
	@echo "  scatter   - run scatter visualization"
	@echo "  pair      - run pair plot visualization"
	@echo "  run_all   - run all visualizations (default 'all' target)"
	@echo "Example: make run_all DATASET=datasets/dataset_test.csv"

clean:
	@echo "Cleaning generated files (reports, images, caches)..."
	# Remove generated files in output/ but keep README.md if present
	@find output -type f ! -name 'README.md' -delete 2>/dev/null || true
	# Remove any empty directories left under output
	@find output -type d -empty -delete 2>/dev/null || true
	# Remove generated images in both top-level and data_visualization subfolders
	@find histograms scatter_plots pair_plot data_visualization/histograms data_visualization/scatter_plots data_visualization/pair_plot -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.pdf' -o -iname '*.svg' \) -delete 2>/dev/null || true
	# Remove python cache directories
	@find . -type d -name '__pycache__' -prune -exec rm -rf {} + || true
	@echo "Clean complete."
