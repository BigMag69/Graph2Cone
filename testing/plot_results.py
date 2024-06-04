import ranx
from ranx import Qrels, Run, evaluate
import torch.utils
from data6 import GraphDatasetEdgesAndNodesWithNegativeSamplesTest, HarderGraphDatasetEdgesAndNodesWithNegativeSamplesTest
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

#results_folder = "results_by_query_size_harder" #"results_harder"
results_folder = "results_by_query_size"

#list all files in results
#results_folder = "results"
#results_folder = "results_by_query_size" #"results_harder"
import os
metrics=["ndcg", "mrr", "ndcg@1", "mrr@5", "precision@5", "recall@5"]
models = ["jacc_baseline", "Graph2Cone14", "Graph2Cone2", "Graph2Cone1", "GIN_baseline",]
model_dicts = {model: {metric: [] for metric in metrics} for model in models}

for q_size in range(1, 11):
    qrels_file = results_folder + f"/qrels{q_size}.json"
    qrels = Qrels.from_file(qrels_file)
    runs = []
    for model in models:
        file = results_folder + f"/{model}{q_size}.json"
        run = Run.from_file(file, name=model)
        runs.append(run)
    report = ranx.compare(qrels=qrels,
                 runs=runs,
                 metrics=metrics,
                 make_comparable=True)
    report.save(results_folder + f"/report{q_size}.json")
    model_dict = report.to_dict()
    for model in models:
        for metric in metrics:
            model_dicts[model][metric].append(model_dict[model]["scores"][metric])



from matplotlib import pyplot as plt

data = model_dicts

# Set up the number of subplots (rows and columns)
n_metrics = len(metrics)
colors = ["#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]

# Set up the number of subplots (rows and columns)
cols = 2
rows = 3

# Create a figure and axes
fig, axes = plt.subplots(rows, cols, figsize=(8, 9))
fig.subplots_adjust(hspace=0.3, wspace=0.2)

# Plot each metric
for idx, metric in enumerate(metrics):
    ax = axes[idx // cols, idx % cols]
    for model_idx, model in enumerate(models):
        model_name = model
        if model == "Graph2Cone14":
            model_name = "Graph2Cone3"
        ax.plot(range(1, 11), data[model][metric], label=model_name, color=colors[model_idx % len(colors)])
        #add "x" marks
        ax.plot(range(1, 11), data[model][metric], 'x', color=colors[model_idx % len(colors)])
    ax.set_xlabel('Number of Edges')
    ax.set_ylabel(metric)

# Hide any empty subplots
for i in range(n_metrics, rows * cols):
    fig.delaxes(axes.flatten()[i])

# Create a shared legend
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=10, frameon=True, framealpha=1, fancybox=True, borderpad=0.5)
#fig.suptitle('Model Performance by Number of Queries (Easier)')

# Adjust layout to remove the gap
#plt.tight_layout(rect=[0, 0, 1, 1])

# save the plot
plt.savefig(results_folder + "/plot.png")