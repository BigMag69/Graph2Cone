import ranx
from ranx import Qrels, Run, evaluate
import torch.utils
from data6 import GraphDatasetEdgesAndNodesWithNegativeSamplesTest, HarderGraphDatasetEdgesAndNodesWithNegativeSamplesTest, HardestGraphDatasetEdgesAndNodesWithNegativeSamplesTest
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os.path as osp
import os
from time import time

device = "cuda" if torch.cuda.is_available() else "cpu"

test_dataset = GraphDatasetEdgesAndNodesWithNegativeSamplesTest(root="graph_dataset_edges_and_nodes_negative_samples_test")[:1000]
test_dataset_harder = HarderGraphDatasetEdgesAndNodesWithNegativeSamplesTest(root="harder_graph_dataset_edges_and_nodes_negative_samples_test")[:1000]
test_dataset_larger = GraphDatasetEdgesAndNodesWithNegativeSamplesTest(root="graph_dataset_edges_and_nodes_negative_samples_test_large")[:1000]
test_dataset_harder_larger = HarderGraphDatasetEdgesAndNodesWithNegativeSamplesTest(root="harder_graph_dataset_edges_and_nodes_negative_samples_test_large")[:1000]
test_dataset_hardest = HardestGraphDatasetEdgesAndNodesWithNegativeSamplesTest(root="hardest_graph_dataset_edges_and_nodes_negative_samples_test", min_edges=7, max_edges=11)[:1000]

class WithoutTheTextGraphs(Dataset):
    def __init__(self, data):
        self.data = data
        super().__init__()

    def len(self):
        return len(self.data)

    def get(self, idx):
        # Extract only the PyTorch graphs, image_id, boolean label, and data index from each tuple
        q2, q1, _, _, negatives, image_idx, data_idx = self.data[idx]
        negatives = [(g, i) for g, _, i in negatives]
        #print(type(q2), type(q1), type(image_idx), type(q_type), type(data_idx))
        return q2, q1, negatives, image_idx, data_idx
    
without_the_text_graphs = WithoutTheTextGraphs(test_dataset)
without_the_text_graphs_harder = WithoutTheTextGraphs(test_dataset_harder)
without_the_text_graphs_larger = WithoutTheTextGraphs(test_dataset_larger)
without_the_text_graphs_harder_larger = WithoutTheTextGraphs(test_dataset_harder_larger)
without_the_text_graphs_hardest = WithoutTheTextGraphs(test_dataset_hardest)
test_loader = DataLoader(without_the_text_graphs, batch_size=1, shuffle=False)
test_loader_harder = DataLoader(without_the_text_graphs_harder, batch_size=1, shuffle=False)
test_loader_larger = DataLoader(without_the_text_graphs_larger, batch_size=1, shuffle=False)
test_loader_harder_larger = DataLoader(without_the_text_graphs_harder_larger, batch_size=1, shuffle=False)
test_loader_hardest = DataLoader(without_the_text_graphs_hardest, batch_size=1, shuffle=False)
results_folder = "resulst_hyper" #"results_harder"

# create the results folder if it does not exist
if not osp.exists(results_folder):
    os.makedirs(results_folder)

min_edges = 1
max_edges = 11
diff = max_edges - min_edges
""" result = {}
for data in [test_dataset_harder, test_dataset_harder_larger]:
    for q_id in tqdm(range(len(data)), total=len(data)):
        query_dict = {}
        _, _, query, graph, negatives, image_idx, data_idx  = data[q_id]
        graphs = [(graph, image_idx)] + [(g, i) for i, (_, g, _) in enumerate(negatives)]
        for i, (g, idx) in enumerate(graphs):
            if i == 0:
                query_dict[str(idx)] = 1
            else:
                query_dict[str(idx)] = 0
        result[str(q_id)] = query_dict

qrels = Qrels(result, name="ground_truths")
qrels.save(results_folder + f"/qrels.json") """


#baselines:
def jacc_baseline():
    def jaccard_distance(set1, set2):
        intersection = set1 & set2 
        union = set1 | set2
        if len(union) == 0:
            return 0
        return len(intersection) / len(union)

    def jaccard_distance2(set1, set2):
        intersection = set1 & set2 
        union = set1 | set2
        if len(union) == 0:
            return 0
        return len(intersection) / len(set1)
    q_rels = {}
    for dataset in [test_dataset_harder, test_dataset_harder_larger]:
        n = len(dataset)
        for q_id in tqdm(range(n), total=n):
            query_dict = {}
            _, _, query, graph, negatives, image_idx, _  = dataset[q_id]
            q_node_map = {i: c for i, c in query.nodes(data="classes")}
            q_edge_labels = [f"{q_node_map[f]} {edge_attr} {q_node_map[t]}"  for f, t, edge_attr in query.edges(data="classes")]
            query_set = set(q_edge_labels)
            graphs = [(graph, image_idx)] + [(g, i) for i, (_, g, _) in enumerate(negatives)]
            for g, idx in graphs:
                g_node_map = {i: c for i, c in g.nodes(data="classes")}
                g_edge_labels = [f"{g_node_map[f]} {edge_attr} {g_node_map[t]}"  for f, t, edge_attr in g.edges(data="classes")]
                graph_set = set(g_edge_labels)
                jacc = jaccard_distance(query_set, graph_set)
                query_dict[str(idx)] = jacc
            tie_breakers = np.linspace(0, 1e-10, len(graphs), endpoint=False)
            np.random.shuffle(tie_breakers)
            for i, idx in enumerate(query_dict):
                query_dict[idx] += tie_breakers[i]

            q_rels[str(q_id)] = query_dict
    return q_rels

#jacc = jacc_baseline()

#run1 = Run(jacc, name="jacc_baseline")
#run1.save(results_folder + f"/jacc_baseline.json")


def sine_halves(theta):
    return torch.sin(theta / 2)

def d_outside(V_q1, V_q2):
    theta_ax1, ap1 = V_q1
    theta_ax2, ap2 = V_q2
    theta_ap1 = ap1 / 2
    theta_ap2 = ap2 / 2
    
    theta_L1 = theta_ax1 - theta_ap1
    theta_U1 = theta_ax1 + theta_ap1
    theta_L2 = theta_ax2 - theta_ap2
    theta_U2 = theta_ax2 + theta_ap2

    distanceU2axis = torch.abs(sine_halves(theta_U1 - theta_ax2))
    distanceL2axis = torch.abs(sine_halves(theta_L1 - theta_ax2))
    distanceax2axis = torch.abs(sine_halves(theta_ax1 - theta_ax2))
    distance_base = torch.abs(sine_halves(theta_ap2))

    indicator_U_in = distanceU2axis < distance_base
    indicator_L_in = distanceL2axis < distance_base
    indicator_axis_in = distanceax2axis < distance_base

    distance_U_out = torch.min(torch.abs(sine_halves(theta_U1 - theta_U2)), torch.abs(sine_halves(theta_U1 - theta_L2)))
    distance_L_out = torch.min(torch.abs(sine_halves(theta_L1 - theta_L2)), torch.abs(sine_halves(theta_L1 - theta_U2)))
    distance_axis_out = torch.min(torch.abs(sine_halves(theta_ax1 - theta_L2)), torch.abs(sine_halves(theta_ax1 - theta_U2)))

    distance_U_out[indicator_U_in] = 0
    distance_L_out[indicator_L_in] = 0
    distance_axis_out[indicator_axis_in] = 0

    distance_out = distance_U_out + distance_L_out + distance_axis_out
    return distance_out

def d_outside2(V_q1, V_q2):
    theta_ax1, ap1 = V_q1
    theta_ax2, ap2 = V_q2
    theta_ap1 = ap1 / 2
    theta_ap2 = ap2 / 2
    
    theta_L2 = theta_ax2 - theta_ap2
    theta_U2 = theta_ax2 + theta_ap2

    distanceax2axis = torch.abs(sine_halves(theta_ax1 - theta_ax2))
    distance_base = torch.abs(sine_halves(theta_ap2))

    indicator_axis_in = distanceax2axis < distance_base
    distance_axis_out = torch.min(torch.abs(sine_halves(theta_ax1 - theta_L2)), torch.abs(sine_halves(theta_ax1 - theta_U2)))
    distance_axis_out[indicator_axis_in] = 0

    distance_out = distance_axis_out
    distance = distance_out
    return distance

def cone_predictions(model, dataloader, answer_model=None, is_query=True):
    # Ensure the model is in eval mode
    model.eval()
    if answer_model is not None:
        answer_model.eval()
    
    results = {}
    with torch.no_grad():
        for dataloader in [test_loader, test_loader_larger]:
            for q2, q1, negatives, img_idx, data_idx in tqdm(dataloader):
                query_dict = {}
                q2 = q2.to(device)
                q1 = q1.to(device)
                query_size = q2.edge_index.size(1)

                gs = [(q1, img_idx)] + [(g.to(device), torch.tensor([i])) for i, (g, _) in enumerate(negatives)] 
                coneb = model(q2.x, q2.edge_index, q2.edge_attr, q2.batch)
                for g, i in gs:
                    if answer_model is not None:
                        conea = answer_model(g.x, g.edge_index, g.edge_attr, g.batch)
                        distance_out = d_outside2(conea, coneb)
                    elif not is_query:
                        conea = model(g.x, g.edge_index, g.edge_attr, g.batch, is_query=False)
                        distance_out = d_outside2(conea, coneb)
                    else:
                        conea = model(g.x, g.edge_index, g.edge_attr, g.batch)
                        distance_out = d_outside(conea, coneb)
                    fully_inside = torch.sum(distance_out <= 0).item()
                    query_dict[str(i.item())] = fully_inside / conea[0].shape[1]
                
                tie_breakers = np.linspace(0, 1e-10, len(gs), endpoint=False)
                np.random.shuffle(tie_breakers)
                for i, idx in enumerate(query_dict):
                    query_dict[idx] += tie_breakers[i]
                results[str(data_idx.item())] = query_dict
        return results

def rate_my_model(model_path, key, test_loader, answer_model_path=None, is_query=False):
    model = torch.load(model_path, map_location=torch.device(device))
    if answer_model_path is not None:
        answer_model = torch.load(answer_model_path, map_location=torch.device(device))
    else:
        answer_model = None
    cone  = cone_predictions(model, test_loader, answer_model, is_query)
    run = Run(cone, name=f"Graph2Cone{key}")
    run.save( results_folder + f"/Graph2Cone{key}.json")

models = {#"k03_gamma05_lambda099": {"path": "hyper_search/Graph2Cone3_k3_gamma5_lambda0.99_small_full_model_2024-06-02_17-11-29.pt",
          #            "is_query": False,
          #            "answer_model_path": None},
          #"k03_gamma05_lambda050": {"path": "hyper_search/Graph2Cone3_k3_gamma5_lambda0.5_small_full_model_2024-06-02_17-05-17.pt",
          #            "is_query": False,
          #            "answer_model_path": None},
          #"k03_gamma05_lambda001": {"path": "hyper_search/Graph2Cone3_k3_gamma5_lambda0.01_small_full_model_2024-06-02_17-09-37.pt",
          #            "is_query": False,
          #            "answer_model_path": None},
          #"k07_gamma05_lambda050": {"path": "hyper_search/Graph2Cone3_k7_gamma5_lambda0.5_small_full_model_2024-06-02_17-47-45.pt",
          #            "is_query": False,
          #            "answer_model_path": None},
          #"k07_gamma05_lambda001": {"path": "hyper_search/Graph2Cone3_k7_gamma5_lambda0.01_small_full_model_2024-06-02_17-56-18.pt",
          #            "is_query": False,
          #            "answer_model_path": None},
          #"k14_gamma05_lambda099": {"path": "hyper_search/Graph2Cone3_k14_gamma5_lambda0.99_small_full_model_2024-06-02_18-54-41.pt",
          #            "is_query": False,
          #            "answer_model_path": None},
          #"k03_gamma10_lambda099": {"path": "hyper_search/Graph2Cone3_k3_gamma10_lambda0.99_small_full_model_2024-06-02_21-17-33.pt",
          #            "is_query": False,
          #            "answer_model_path": None},
          #"k07_gamma10_lambda099": {"path": "hyper_search/Graph2Cone3_k7_gamma10_lambda0.99_small_full_model_2024-06-02_21-38-27.pt",
          #            "is_query": False,
          #            "answer_model_path": None},
          #"k14_gamma05_lambda050": {"path": "hyper_search/Graph2Cone3_k14_gamma5_lambda0.5_small_full_model_2024-06-02_19-34-25.pt",
          #            "is_query": False,
          #            "answer_model_path": None},
          #"k03_gamma10_lambda050": {"path": "hyper_search/Graph2Cone3_k3_gamma10_lambda0.5_small_full_model_2024-06-02_22-57-38.pt",
          #            "is_query": False,
          #            "answer_model_path": None},
          #"k14_gamma10_lambda099": {"path": "hyper_search/Graph2Cone3_k14_gamma10_lambda0.99_small_full_model_2024-06-02_22-23-58.pt",
          #            "is_query": False,
          #            "answer_model_path": None},
          #"k03_gamma20_lambda099": {"path": "hyper_search/Graph2Cone3_k3_gamma20_lambda0.99_small_full_model_2024-06-02_23-45-02.pt",
          #            "is_query": False,
          #            "answer_model_path": None},
          #"k07_gamma10_lambda050": {"path": "hyper_search/Graph2Cone3_k7_gamma10_lambda0.5_small_full_model_2024-06-02_23-20-59.pt",
          #            "is_query": False,
          #            "answer_model_path": None},
          #"k07_gamma20_lambda099": {"path": "hyper_search/Graph2Cone3_k7_gamma20_lambda0.99_small_full_model_2024-06-02_23-58-17.pt",
          #            "is_query": False,
          #            "answer_model_path": None},
          #"k14_gamma20_lambda099": {"path": "hyper_search/Graph2Cone3_k14_gamma20_lambda0.99_small_full_model_2024-06-03_00-26-30.pt",
          #            "is_query": False,
          #            "answer_model_path": None},
          #"k14_gamma10_lambda050": {"path": "hyper_search/Graph2Cone3_k14_gamma10_lambda0.5_small_full_model_2024-06-03_00-24-03.pt",
          #            "is_query": False,
          #            "answer_model_path": None},
          #"k14_gamma05_lambda001": {"path": "hyper_search/Graph2Cone3_k14_gamma5_lambda0.01_small_full_model_2024-06-02_20-25-32.pt",
          #            "is_query": False,
          #            "answer_model_path": None},
          #"k03_gamma20_lambda050": {"path": "hyper_search/Graph2Cone3_k3_gamma20_lambda0.5_small_full_model_2024-06-03_02-04-28.pt",
          #            "is_query": False,
          #            "answer_model_path": None},
          #"k03_gamma05_lambda090": {"path": "hyper_search/Graph2Cone3_k3_gamma5_lambda0.9_small_full_model_2024-06-03_01-57-05.pt",
          #            "is_query": False,
          #            "answer_model_path": None},
          "k03_gamma10_lambda001": {"path": "hyper_search/Graph2Cone3_k3_gamma10_lambda0.01_small_full_model_2024-06-03_02-28-07.pt",
                      "is_query": False,
                      "answer_model_path": None},
          "k07_gamma05_lambda090": {"path": "hyper_search/Graph2Cone3_k7_gamma5_lambda0.9_small_full_model_2024-06-03_02-28-29.pt",
                      "is_query": False,
                      "answer_model_path": None},
          "k03_gamma20_lambda050": {"path": "hyper_search/Graph2Cone3_k7_gamma20_lambda0.5_small_full_model_2024-06-03_02-22-55.pt",
                      "is_query": False,
                      "answer_model_path": None},
          "k14_gamma05_lambda090": {"path": "hyper_search/Graph2Cone3_k14_gamma5_lambda0.9_small_full_model_2024-06-03_03-36-34.pt",
                      "is_query": False,
                      "answer_model_path": None},
          "k03_gamma10_lambda090": {"path": "hyper_search/Graph2Cone3_k3_gamma10_lambda0.9_small_full_model_2024-06-03_05-41-08.pt",
                      "is_query": False,
                      "answer_model_path": None},
          "k07_gamma10_lambda001": {"path": "hyper_search/Graph2Cone3_k7_gamma10_lambda0.01_small_full_model_2024-06-03_03-02-35.pt",
                      "is_query": False,
                      "answer_model_path": None},
          "k07_gamma10_lambda090": {"path": "hyper_search/Graph2Cone3_k7_gamma10_lambda0.9_small_full_model_2024-06-03_06-00-06.pt",
                      "is_query": False,
                      "answer_model_path": None},
          "k14_gamma20_lambda050": {"path": "hyper_search/Graph2Cone3_k14_gamma20_lambda0.5_small_full_model_2024-06-03_04-28-40.pt",
                      "is_query": False,
                      "answer_model_path": None},
          "k14_gamma10_lambda090": {"path": "hyper_search/Graph2Cone3_k14_gamma10_lambda0.9_small_full_model_2024-06-03_06-45-43.pt",
                      "is_query": False,
                      "answer_model_path": None},
          "k03_gamma20_lambda090": {"path": "hyper_search/Graph2Cone3_k3_gamma20_lambda0.9_small_full_model_2024-06-03_07-57-25.pt",
                      "is_query": False,
                      "answer_model_path": None},
          "k07_gamma2o_lambda090": {"path": "hyper_search/Graph2Cone3_k7_gamma20_lambda0.9_small_full_model_2024-06-03_08-16-35.pt",
                      "is_query": False,
                      "answer_model_path": None},
          "k14_gamma20_lambda090": {"path": "hyper_search/Graph2Cone3_k14_gamma20_lambda0.9_small_full_model_2024-06-03_08-56-35.pt",
                      "is_query": False,
                      "answer_model_path": None},
          "k03_gamma05_lambda010": {"path": "hyper_search/Graph2Cone3_k3_gamma5_lambda0.1_small_full_model_2024-06-03_10-56-59.pt",
                      "is_query": False,
                      "answer_model_path": None},
          "k14_gamma10_lambda001": {"path": "hyper_search/Graph2Cone3_k14_gamma10_lambda0.01_small_full_model_2024-06-03_06-33-28.pt",
                      "is_query": False,
                      "answer_model_path": None},
          "k07_gamma05_lambda010": {"path": "hyper_search/Graph2Cone3_k7_gamma5_lambda0.1_small_full_model_2024-06-03_11-38-23.pt",
                      "is_query": False,
                      "answer_model_path": None},
          "k03_gamma20_lambda001": {"path": "hyper_search/Graph2Cone3_k3_gamma20_lambda0.01_small_full_model_2024-06-03_12-51-38.pt",
                      "is_query": False,
                      "answer_model_path": None},
          "k14_gamma20_lambda010": {"path": "hyper_search/Graph2Cone3_k14_gamma20_lambda0.1_small_full_model_2024-06-03_07-51-18.pt",
                      "is_query": False,
                      "answer_model_path": None},           
          "k14_gamma05_lambda010": {"path": "hyper_search/Graph2Cone3_k14_gamma5_lambda0.1_small_full_model_2024-06-03_13-54-42.pt",
                      "is_query": False,
                      "answer_model_path": None},     
          "k07_gamma20_lambda001": {"path": "hyper_search/Graph2Cone3_k7_gamma20_lambda0.01_small_full_model_2024-06-03_14-17-15.pt",
                      "is_query": False,
                      "answer_model_path": None},  
          "k03_gamma10_lambda010": {"path": "hyper_search/Graph2Cone3_k3_gamma10_lambda0.1_small_full_model_2024-06-03_17-18-12.pt",
                      "is_query": False,
                      "answer_model_path": None},  
          "k07_gamma20_lambda010": {"path": "hyper_search/Graph2Cone3_k7_gamma20_lambda0.1_small_full_model_2024-06-03_16-32-55.pt",
                      "is_query": False,
                      "answer_model_path": None},  
          "k07_gamma10_lambda010": {"path": "hyper_search/Graph2Cone3_k7_gamma10_lambda0.1_small_full_model_2024-06-03_18-04-11.pt",
                      "is_query": False,
                      "answer_model_path": None},  
          "k14_gamma20_lambda001": {"path": "hyper_search/Graph2Cone3_k14_gamma20_lambda0.01_small_full_model_2024-06-03_17-48-52.pt",
                      "is_query": False,
                      "answer_model_path": None},  
          "k07_gamma05_lambda099": {"path": "hyper_search/Graph2Cone3_k7_gamma5_lambda0.99_small_full_model_2024-06-02_17-43-54.pt",
                      "is_query": False,
                      "answer_model_path": None},  
          "k14_gamma10_lambda010": {"path": "hyper_search/Graph2Cone3_k14_gamma10_lambda0.1_small_full_model_2024-06-03_21-03-32.pt",
                      "is_query": False,
                      "answer_model_path": None},  
          "k03_gamma20_lambda010": {"path": "hyper_search/Graph2Cone3_k3_gamma20_lambda0.1_small_full_model_2024-06-03_20-00-12.pt",
                      "is_query": False,
                      "answer_model_path": None}
}

for key, value in models.items():
    print(value)
    rate_my_model(value["path"], key, test_loader, value["answer_model_path"], value["is_query"])