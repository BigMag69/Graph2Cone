import torch
from sklearn.metrics import precision_recall_fscore_support
from torch_geometric.loader import DataLoader
#from baseline_models import GINSubgraphModel
from models import Graph2Cone, cone_loss, d_con3
from tqdm.auto import tqdm

# Plotting functions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import cm


def cone_to_wedge(axis, aperture, color):
    #convert to degrees
    theta1 = axis - aperture/2
    theta2 = axis + aperture/2
    theta1 = np.degrees(theta1)
    theta2 = np.degrees(theta2)
    #t1, t2 = min(theta1, theta2), max(theta1, theta2)
    return patches.Wedge((0, 0), 1, theta1, theta2, fill=True, color=color, linewidth=0.5, alpha=0.5)

def draw_cones(cones, dim, ax=None, right=True):
    if ax is None:
        fig, ax = plt.subplots()

    colors = ["cornflowerblue", "indianred"]
    circle_color = 'black' if right else 'red'
    # plot the unit circle
    circle = patches.Circle((0, 0), radius=1, fill=False, edgecolor=circle_color, linewidth=0.5)
    ax.add_patch(circle)

    # Plot the first cone
    wedges = []
    for color, cone in zip(colors, cones):
        wedge = cone_to_wedge(cone[0], cone[1], color=color)
        ax.add_patch(wedge)
        wedges.append(wedge)

    ax.set_xlim(-1,1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.set_title(f'{dim}', color=circle_color)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return wedges

def plot_cone_grid(queries, graphs):
    fig, axes = plt.subplots(nrows=16, ncols=16, figsize=(20, 20))

    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        #print((queries[0][0][idx], queries[1][0][idx]))
        #print((graphs[0][0][idx], graphs[1][0][idx]))
        wedges = draw_cones([(queries[0][0][idx], queries[1][0][idx]), (graphs[0][0][idx], graphs[1][0][idx])], dim=idx, ax=ax, right=True)

    fig.legend(wedges, ['Query', 'Graph'], loc='upper right', bbox_to_anchor=(1, 1), title='Cones')
    fig.suptitle('Cone embeddings for query and graph', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()



if __name__ == '__main__':
    from basic_data import GraphDatasetBestBlip2CleanedSentencesOnEdgesPreprocessed
    from largest_cc_data import GraphDatasetLargestConnectedComponentWithNegativeSamples

    # Load the data
    basic_data = GraphDatasetBestBlip2CleanedSentencesOnEdgesPreprocessed(root="graph_dataset_best_blip_cleaned_sentences_on_edges_preprocessed")
    dataset = GraphDatasetLargestConnectedComponentWithNegativeSamples(root="graph_dataset_largest_connected_component_with_negative_samples", basic_data=basic_data)

    # Load the model
    path = "Graph2Cone_saved_files/Graph2Cone_full_model_2024-04-23_10-34-04.pt"
    model = torch.load(path, map_location=torch.device('cpu'))

    test_dataset = dataset[45783:]
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Get the cones
    model.eval()
    with torch.no_grad():
        for q2, q1, negative_sampling, idx in test_loader:

            # Get the cones
            query = model(q2.x, q2.edge_index, q2.edge_attr, q2.batch)
            graph = model(q1.x, q1.edge_index, q1.edge_attr, q1.batch)
            # Plot the cones
            #print(query)
            plot_cone_grid(query, graph)
            break
