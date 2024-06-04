import torch
from torch_geometric.data import Dataset
from tqdm import tqdm
import os.path as osp
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils.convert import to_networkx
import networkx as nx
from util import *
import pickle
import random
from basic_data import GraphDatasetBasic
import json
from collections import deque
from itertools import permutations, combinations
#import networkx.algorithms.isomorphism as iso
import networkx.algorithms.isomorphism as iso


device = "cuda" if torch.cuda.is_available() else "cpu"

config_dict = utilfunc_get_config_dict()

image_ids_with_coco_captions_with_nodes = utilfunc_read_pickle_to_list("helper_files/image_ids_with_coco_captions_with_nodes.pickle")

def is_subgraph(main_graph, sub_graph):
    GM = iso.DiGraphMatcher(main_graph, 
                            sub_graph, 
                            node_match=iso.categorical_node_match("classes", None), 
                            edge_match=iso.categorical_edge_match("classes", None))
    return GM.subgraph_is_isomorphic()


class GraphDatasetEdgesAndNodesWithNegativeSamples(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, custom_transform=None, k=7, min_edges=5, max_edges=11):
        self.filename = "graph_info.txt"
        self.name = f"GraphDatasetEdgesAndNodesWithNegativeSamples"
        with open("edge_graph_dict.json", "r") as f:
            self.edge_graph_dict = json.load(f)
        self.basic_data_full = GraphDatasetBasic(root="graph_dataset_basic")
        self.basic_data = self.basic_data_full[:10000] #reserve the last 1000 graphs for testing
        self.k = k #negative sample size
        self.min_edges = min_edges
        self.max_edges = max_edges
        root = f"{root}_{k}"
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        graph_info_file = open(self.raw_paths[0], "r")
        number_of_graphs = int(graph_info_file.read())
        graph_info_file.close()
        return [f'data_{i}.pt' for i in range(number_of_graphs)]

    def download(self):
        pass

    def process(self):
        #if osp.exists(osp.join(self.raw_dir, self.filename)):
        #    return
        idx = 0
        for graph, text_graph, image_idx, data_idx in tqdm(self.basic_data, total=len(self.basic_data)):
            G = to_networkx(graph, node_attrs=["x"], edge_attrs=["edge_attr"])
            text_node_map = {j: i for i, j in enumerate(text_graph.nodes)}
            nx.relabel_nodes(text_graph, text_node_map, copy=False)

            largest_cc = max(nx.weakly_connected_components(G), key=len)
            G_cc = G.subgraph(largest_cc).copy()
            edges = list(G_cc.edges)

            if len(edges) < self.min_edges:
                continue
            for n_edges in range(self.min_edges, min(self.max_edges, len(edges))):
                if osp.exists(osp.join(self.processed_dir, f'data_{idx}.pt')):
                    idx += 1
                    continue
                
                cc_edges = self._get_connected_components(G, n_edges)
                if cc_edges is None:
                    break

                G2 = G.edge_subgraph(cc_edges).copy()
                text_G2 = text_graph.edge_subgraph(cc_edges).copy()
                negative_samples = self._get_negative_samples(text_G2, data_idx)
                if negative_samples is None:
                    break
                pyg_query = from_networkx(G2).to(device)

                if pyg_query.edge_attr is None:
                    pyg_query.edge_attr = torch.zeros(0, 256).to(device)

                tuple = (pyg_query, negative_samples, data_idx, idx)

                torch.save(tuple, osp.join(self.processed_dir, f'data_{idx}.pt'))
                idx += 1
        with open(osp.join(self.raw_dir, self.filename), "w") as f:
            f.write(str(idx))
    
    def _get_connected_components(self, G, n_edges):
        UG = G.to_undirected()

        def bfs(start):
            path_edges = []
            queue = deque([start])  
            visited = set([start])
            
            while queue:
                current = queue.popleft()
                
                if len(path_edges) == n_edges:
                    return path_edges
                neighborhood = list(UG[current])
                random.shuffle(neighborhood)
                for neighbor in neighborhood:
                    edge = (current, neighbor) if (current, neighbor) in G.edges() else (neighbor, current)
                    if edge not in path_edges:
                        path_edges = path_edges + [edge]
                        if len(path_edges) == n_edges:
                            return path_edges
                        if neighbor not in visited or len(path_edges) < n_edges:
                            visited.add(neighbor)
                            queue.append(neighbor)
            return None
        nodes = list(UG.nodes())
        random.shuffle(nodes)
        for node in nodes:
            result = bfs(node)
            if result is not None:
                return result
        return None

    def _get_negative_samples(self, text_G2, data_idx):
        negative_samples = []
        tried_graphs = [data_idx]
        while len(negative_samples) < self.k:
            if len(tried_graphs) == 50783:
                return None
            graph_idx = random.randint(0, 50783-1)
            if graph_idx in tried_graphs:
                continue
            tried_graphs.append(graph_idx)     

            if graph_idx in negative_samples:
                continue
            _, text_graph, _, _ = self.basic_data_full.get(graph_idx)

            if text_graph.number_of_edges() < 1:
                continue

            text_node_map = {j: i for i, j in enumerate(text_graph.nodes)}
            nx.relabel_nodes(text_graph, text_node_map, copy=False)
            if is_subgraph(text_graph, text_G2): 
                continue
            negative_samples.append(graph_idx)
        return negative_samples

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        pyg_query, negative_samples, image_idx, data_idx = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'), map_location=device)
        pyg_graph = self.basic_data_full.get(image_idx)[0]
        negative_samples = [(self.basic_data_full[i][0], i) for i in negative_samples]
        return (pyg_query, pyg_graph, negative_samples, image_idx, data_idx)

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'

class GraphDatasetEdgesAndNodesWithNegativeSamplesTest(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, custom_transform=None, k=20, min_edges=5, max_edges=11):
        self.filename = "graph_info.txt"
        self.name = f"GraphDatasetEdgesAndNodesWithNegativeSamplesTest"
        with open("edge_graph_dict.json", "r") as f:
            self.edge_graph_dict = json.load(f)
        self.basic_data_full = GraphDatasetBasic(root="graph_dataset_basic")
        self.basic_data = self.basic_data_full[-5000:] #reserve the last 1000 graphs for testing
        self.k = k #negative sample size
        self.min_edges = min_edges
        self.max_edges = max_edges
        root = root
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        graph_info_file = open(self.raw_paths[0], "r")
        number_of_graphs = int(graph_info_file.read())
        graph_info_file.close()
        return [f'data_{i}.pt' for i in range(number_of_graphs)]

    def download(self):
        pass

    def process(self):
        if self.len() > 0:
            return
        #if osp.exists(osp.join(self.raw_dir, self.filename)):
        #    return
        idx = 0
        for graph, text_graph, image_idx, data_idx in tqdm(self.basic_data, total=len(self.basic_data)):
            G = to_networkx(graph, node_attrs=["x"], edge_attrs=["edge_attr"])
            text_node_map = {j: i for i, j in enumerate(text_graph.nodes)}
            nx.relabel_nodes(text_graph, text_node_map, copy=False)
            
            largest_cc = max(nx.weakly_connected_components(G), key=len)
            G_cc = G.subgraph(largest_cc).copy()
            edges = list(G_cc.edges)

            if len(edges) < self.min_edges:
                continue
            
            for n_edges in range(self.min_edges, self.max_edges):
                if osp.exists(osp.join(self.processed_dir, f'data_{idx}.pt')):
                    idx += 1
                    continue
                
                cc_edges = self._get_connected_components(G, n_edges)
                if cc_edges is None:
                    break

                G2 = G.edge_subgraph(cc_edges).copy()
                text_G2 = text_graph.edge_subgraph(cc_edges).copy()
                negative_samples = self._get_negative_samples(text_G2, data_idx)
                
                pyg_query = from_networkx(G2).to(device)

                if pyg_query.edge_attr is None:
                    pyg_query.edge_attr = torch.zeros(0, 256).to(device)

                tuple = (pyg_query, text_G2, negative_samples, data_idx, idx)
                torch.save(tuple, osp.join(self.processed_dir, f'data_{idx}.pt'))
                idx += 1
        with open(osp.join(self.raw_dir, self.filename), "w") as f:
            f.write(str(idx))
    
    def _get_connected_components(self, G, n_edges):
        UG = G.to_undirected()

        def bfs(start):
            path_edges = []
            queue = deque([start])  
            visited = set([start])
            
            while queue:
                current = queue.popleft()
                
                if len(path_edges) == n_edges:
                    return path_edges
                neighborhood = list(UG[current])
                random.shuffle(neighborhood)
                for neighbor in neighborhood:
                    edge = (current, neighbor) if (current, neighbor) in G.edges() else (neighbor, current)
                    if edge not in path_edges:
                        path_edges = path_edges + [edge]
                        if len(path_edges) == n_edges:
                            return path_edges
                        if neighbor not in visited or len(path_edges) < n_edges:
                            visited.add(neighbor)
                            queue.append(neighbor)
            return None
        nodes = list(UG.nodes())
        random.shuffle(nodes)
        for node in nodes:
            result = bfs(node)
            if result is not None:
                return result
        return None

    def _get_negative_samples(self, text_G2, data_idx):
        negative_samples = []
        while len(negative_samples) < self.k:
            graph_idx = random.randint(45783, 50783-1)

            if graph_idx in negative_samples or graph_idx == data_idx:
                continue
            _, text_graph, _, _ = self.basic_data_full.get(graph_idx)
            
            if text_graph.number_of_edges() < 1:
                continue

            text_node_map = {j: i for i, j in enumerate(text_graph.nodes)}
            nx.relabel_nodes(text_graph, text_node_map, copy=False)
            if is_subgraph(text_graph, text_G2):  
                continue
            negative_samples.append(graph_idx)
        return negative_samples

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        pyg_query, text_G2, negative_samples, image_idx, data_idx = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'), map_location=device)
        pyg_graph = self.basic_data_full.get(image_idx)[0]
        text_graph = self.basic_data_full.get(image_idx)[1]
        negative_samples = [(self.basic_data_full[i][0], self.basic_data_full[i][1], i) for i in negative_samples]
        return (pyg_query, pyg_graph, text_G2, text_graph, negative_samples, image_idx, data_idx)

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'


class HarderGraphDatasetEdgesAndNodesWithNegativeSamplesTest(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, custom_transform=None, k=20, min_edges=5, max_edges=11):
        self.filename = "graph_info.txt"
        self.name = f"HarderGraphDatasetEdgesAndNodesWithNegativeSamplesTest"
        with open("edge_graph_dict.json", "r") as f:
            self.edge_graph_dict = json.load(f)
        self.basic_data_full = GraphDatasetBasic(root="graph_dataset_basic")
        self.basic_data = self.basic_data_full[-5000:] #reserve the last 1000 graphs for testing
        self.k = k #negative sample size
        self.min_edges = min_edges
        self.max_edges = max_edges
        root = root
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        graph_info_file = open(self.raw_paths[0], "r")
        number_of_graphs = int(graph_info_file.read())
        graph_info_file.close()
        return [f'data_{i}.pt' for i in range(number_of_graphs)]

    def download(self):
        pass

    def process(self):
        if self.len() > 0:
            return
        #if osp.exists(osp.join(self.raw_dir, self.filename)):
        #    return
        idx = 0
        for graph, text_graph, image_idx, data_idx in tqdm(self.basic_data, total=len(self.basic_data)):
            G = to_networkx(graph, node_attrs=["x"], edge_attrs=["edge_attr"])
            text_node_map = {j: i for i, j in enumerate(text_graph.nodes)}
            nx.relabel_nodes(text_graph, text_node_map, copy=False)
            largest_cc = max(nx.weakly_connected_components(G), key=len)
            G_cc = G.subgraph(largest_cc).copy()
            edges = list(G_cc.edges)

            if len(edges) < self.min_edges:
                continue
            
            for n_edges in range(self.min_edges, self.max_edges):
                if osp.exists(osp.join(self.processed_dir, f'data_{idx}.pt')):
                    idx += 1
                    continue
                
                cc_edges = self._get_connected_components(G, n_edges)
                if cc_edges is None:
                    break

                G2 = G.edge_subgraph(cc_edges).copy()
                text_G2 = text_graph.edge_subgraph(cc_edges).copy()
                negative_samples = self._get_negative_samples(text_G2, data_idx)
                if negative_samples is None:
                    break
                
                pyg_query = from_networkx(G2).to(device)

                if pyg_query.edge_attr is None:
                    pyg_query.edge_attr = torch.zeros(0, 256).to(device)

                tuple = (pyg_query, text_G2, negative_samples, data_idx, idx)
                torch.save(tuple, osp.join(self.processed_dir, f'data_{idx}.pt'))
                idx += 1
        with open(osp.join(self.raw_dir, self.filename), "w") as f:
            f.write(str(idx))
    
    def _get_connected_components(self, G, n_edges):
        UG = G.to_undirected()

        def bfs(start):
            path_edges = []
            queue = deque([start])  
            visited = set([start])
            
            while queue:
                current = queue.popleft()
                
                if len(path_edges) == n_edges:
                    return path_edges
                neighborhood = list(UG[current])
                random.shuffle(neighborhood)
                for neighbor in neighborhood:
                    edge = (current, neighbor) if (current, neighbor) in G.edges() else (neighbor, current)
                    if edge not in path_edges:
                        path_edges = path_edges + [edge]
                        if len(path_edges) == n_edges:
                            return path_edges
                        if neighbor not in visited or len(path_edges) < n_edges:
                            visited.add(neighbor)
                            queue.append(neighbor)
            return None
        nodes = list(UG.nodes())
        random.shuffle(nodes)
        for node in nodes:
            result = bfs(node)
            if result is not None:
                return result
        return None

    def _get_negative_samples(self, text_G2, data_idx):
        negative_samples = []
        tried_graphs = [data_idx]
        while len(negative_samples) < self.k:
            if len(tried_graphs) == 50783:
                return None
            graph_idx = None
            if len(text_G2.edges) > 1:
                edge_sentences = [f"{text_G2.nodes[f]['classes']} {attr['classes']} {text_G2.nodes[t]['classes']}" for f, t, attr in text_G2.edges(data=True)]
                for sentence in edge_sentences:
                    graph_indices = self.edge_graph_dict.get(sentence)
                    if graph_indices is not None:
                        random.shuffle(graph_indices)
                        for graph_idx in graph_indices:
                            if graph_idx not in tried_graphs:
                                tried_graphs.append(graph_idx)
                                break
                            else:
                                graph_idx = None
                        if graph_idx is not None:
                            break
            if graph_idx is None:
                graph_idx = random.randint(0, 50783-1)
                if graph_idx in tried_graphs:
                    continue
                tried_graphs.append(graph_idx)     

            if graph_idx in negative_samples:
                continue
            _, text_graph, _, _ = self.basic_data_full.get(graph_idx)

            if text_graph.number_of_edges() < 1:
                continue

            text_node_map = {j: i for i, j in enumerate(text_graph.nodes)}
            nx.relabel_nodes(text_graph, text_node_map, copy=False)
            if is_subgraph(text_graph, text_G2): 
                continue
            negative_samples.append(graph_idx)
        return negative_samples

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        pyg_query, text_G2, negative_samples, image_idx, data_idx = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'), map_location=device)
        pyg_graph = self.basic_data_full.get(image_idx)[0]
        text_graph = self.basic_data_full.get(image_idx)[1]
        negative_samples = [(self.basic_data_full.get(i)[0], self.basic_data_full.get(i)[1], i) for i in negative_samples]
        return (pyg_query, pyg_graph, text_G2, text_graph, negative_samples, image_idx, data_idx)

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'
    

def split_edges(original_graph, edges_to_split):
    # Create a copy of the original graph
    new_graph = original_graph.copy()

    # Loop over each edge to split
    for edge in edges_to_split:
        source, target = edge
        
        # Create new node for the source
        new_source = max(new_graph.nodes) + 1
        new_graph.add_node(new_source, **original_graph.nodes[source])
        source_edge_data = original_graph.get_edge_data(source, target)
        new_graph.add_edge(new_source, target, **source_edge_data)
        new_graph.remove_edge(source, target)
        
        # Create new node for the target
        new_target = max(new_graph.nodes) + 1
        new_graph.add_node(new_target, **original_graph.nodes[target])
        new_graph.add_edge(new_source, new_target, **source_edge_data)
        new_graph.remove_edge(new_source, target)
        
    # Remove nodes without edges
    isolated_nodes = list(nx.isolates(new_graph))
    new_graph.remove_nodes_from(isolated_nodes)
    
    # Reindex the nodes to go from 0 to n
    mapping = {old_label: new_label for new_label, old_label in enumerate(new_graph.nodes())}
    new_graph = nx.relabel_nodes(new_graph, mapping)
    
    return new_graph    

def generate_edge_combinations(graph, min_edges):
    edges = list(graph.edges())
    edge_combinations = []
    
    for r in range(min_edges, len(edges) + 1):
        combinations_r = list(combinations(edges, r))
        edge_combinations.extend(combinations_r)
    
    return edge_combinations


class HardestGraphDatasetEdgesAndNodesWithNegativeSamplesTest(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, custom_transform=None, k=20, min_edges=7, max_edges=11):
        self.filename = "graph_info.txt"
        self.name = f"HardestGraphDatasetEdgesAndNodesWithNegativeSamplesTest"
        with open("edge_graph_dict.json", "r") as f:
            self.edge_graph_dict = json.load(f)
        self.basic_data_full = GraphDatasetBasic(root="graph_dataset_basic")
        self.basic_data = self.basic_data_full[-5000:-4000] #reserve the last 1000 graphs for testing
        self.k = k #negative sample size
        self.min_edges = min_edges
        self.max_edges = max_edges
        root = root
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        graph_info_file = open(self.raw_paths[0], "r")
        number_of_graphs = int(graph_info_file.read())
        graph_info_file.close()
        return [f'data_{i}.pt' for i in range(number_of_graphs)]

    def download(self):
        pass

    def process(self):
        if self.len() > 0:
            return
        #if osp.exists(osp.join(self.raw_dir, self.filename)):
        #    return
        idx = 0
        for graph, text_graph, image_idx, data_idx in tqdm(self.basic_data, total=len(self.basic_data)):
            G = to_networkx(graph, node_attrs=["x"], edge_attrs=["edge_attr"])
            text_node_map = {j: i for i, j in enumerate(text_graph.nodes)}
            nx.relabel_nodes(text_graph, text_node_map, copy=False)
            largest_cc = max(nx.weakly_connected_components(G), key=len)
            G_cc = G.subgraph(largest_cc).copy()
            edges = list(G_cc.edges)

            if len(edges) < self.min_edges:
                continue
            
            for n_edges in range(self.min_edges, self.max_edges):
                if osp.exists(osp.join(self.processed_dir, f'data_{idx}.pt')):
                    idx += 1
                    continue
                
                cc_edges = self._get_connected_components(G, n_edges)
                if cc_edges is None:
                    break

                G2 = G.edge_subgraph(cc_edges).copy()
                text_G2 = text_graph.edge_subgraph(cc_edges).copy()
                negative_samples = self._get_negative_samples(text_G2, G, text_graph, data_idx)
                if negative_samples is None:
                    break

                # Remove nodes without edges
                isolated_nodes = list(nx.isolates(G))
                G.remove_nodes_from(isolated_nodes)
                text_graph.remove_nodes_from(isolated_nodes)
                                
                pyg_query = from_networkx(G2).to(device)

                if pyg_query.edge_attr is None:
                    pyg_query.edge_attr = torch.zeros(0, 256).to(device)

                tuple = (from_networkx(G).to(device), text_graph, pyg_query, text_G2, negative_samples, data_idx, idx)
                torch.save(tuple, osp.join(self.processed_dir, f'data_{idx}.pt'))
                idx += 1
        with open(osp.join(self.raw_dir, self.filename), "w") as f:
            f.write(str(idx))
    
    def _get_connected_components(self, G, n_edges):
        UG = G.to_undirected()

        def bfs(start):
            path_edges = []
            queue = deque([start])  
            visited = set([start])
            
            while queue:
                current = queue.popleft()
                
                if len(path_edges) == n_edges:
                    return path_edges
                neighborhood = list(UG[current])
                random.shuffle(neighborhood)
                for neighbor in neighborhood:
                    edge = (current, neighbor) if (current, neighbor) in G.edges() else (neighbor, current)
                    if edge not in path_edges:
                        path_edges = path_edges + [edge]
                        if len(path_edges) == n_edges:
                            return path_edges
                        if neighbor not in visited or len(path_edges) < n_edges:
                            visited.add(neighbor)
                            queue.append(neighbor)
            return None
        nodes = list(UG.nodes())
        random.shuffle(nodes)
        for node in nodes:
            result = bfs(node)
            if result is not None:
                return result
        return None

    def _get_negative_samples(self, text_G2, graph, text_graph, data_idx):
        negative_samples = []
        splits = generate_edge_combinations(text_G2, min_edges=5)
        if len(splits) < self.k:
            return None
        #choose k random splits
        random.shuffle(splits)
        splits = splits[:self.k]
        for split in splits:
            new_graph = split_edges(graph, split)
            new_text_graph = split_edges(text_graph, split)
            negative_samples.append((from_networkx(new_graph).to(device), new_text_graph, -1))
        return negative_samples

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        pyg_graph, text_graph, pyg_query, text_G2, negative_samples, image_idx, data_idx = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'), map_location=device)
        return (pyg_query, pyg_graph, text_G2, text_graph, negative_samples, image_idx, data_idx)

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'


#dataset = GraphDatasetEdgesAndNodesWithNegativeSamples()
#test_dataset = GraphDatasetEdgesAndNodesWithNegativeSamplesTest()
#harder_test_dataset = HarderGraphDatasetEdgesAndNodesWithNegativeSamplesTest()
if __name__ == "__main__":
    print("small")
    test_dataset = GraphDatasetEdgesAndNodesWithNegativeSamplesTest(root="graph_dataset_edges_and_nodes_negative_samples_test", min_edges=1, max_edges=5)
    print("small harder")
    test_dataset_harder = HarderGraphDatasetEdgesAndNodesWithNegativeSamplesTest(root="harder_graph_dataset_edges_and_nodes_negative_samples_test", min_edges=1, max_edges=5)
    print("large")
    test_dataset_larger = GraphDatasetEdgesAndNodesWithNegativeSamplesTest(root="graph_dataset_edges_and_nodes_negative_samples_test_large", min_edges=5, max_edges=11)
    print("large harder")
    test_dataset_harder_larger = HarderGraphDatasetEdgesAndNodesWithNegativeSamplesTest(root="harder_graph_dataset_edges_and_nodes_negative_samples_test_large", min_edges=5, max_edges=11)
    print("hardest")
    test_dataset_hardest = HardestGraphDatasetEdgesAndNodesWithNegativeSamplesTest(root="hardest_graph_dataset_edges_and_nodes_negative_samples_test", min_edges=7, max_edges=11)
    print("train")
    train_dataset = GraphDatasetEdgesAndNodesWithNegativeSamples(root="graph_dataset_edges_and_nodes_negative_samples_train", min_edges=5, max_edges=11, k=3)
    train_dataset2 = GraphDatasetEdgesAndNodesWithNegativeSamples(root="graph_dataset_edges_and_nodes_negative_samples_train", min_edges=5, max_edges=11, k=7)
    train_dataset3 = GraphDatasetEdgesAndNodesWithNegativeSamples(root="graph_dataset_edges_and_nodes_negative_samples_train", min_edges=5, max_edges=11, k=14)