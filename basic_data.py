import torch
from torch_geometric.data import Dataset
from tqdm import tqdm
import os.path as osp
from torch_geometric.utils.convert import from_networkx
import networkx as nx
from util import *
import pickle
from blip_utils import load_default_blip2_model


config_dict = utilfunc_get_config_dict()

training_set_end_index = config_dict["training_set_end_index"]

device = "cuda" if torch.cuda.is_available() else "cpu"
blip_model, preprocess = load_default_blip2_model(device=device)

image_ids_with_coco_captions_with_nodes = utilfunc_read_pickle_to_list("helper_files/image_ids_with_coco_captions_with_nodes.pickle")

class GraphDatasetBasic(Dataset):
    def __init__(self, root="graph_dataset_basic", transform=None, pre_transform=None, pre_filter=None, custom_transform=None):
        self.filename = "graph_info.txt"
        self.name = "GraphDatasetBasic"
        super().__init__(root, transform, pre_transform, pre_filter)
        self.custom_transform = custom_transform
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        if not osp.exists(osp.join(self.raw_dir, self.filename)):
            with open(osp.join(self.raw_dir, self.filename), "w") as f:
                f.write(str(len(image_ids_with_coco_captions_with_nodes)))
        graph_info_file = open(self.raw_paths[0], "r")
        number_of_graphs = int(graph_info_file.read())
        graph_info_file.close()
        return [f'data_{i}.pt' for i in range(number_of_graphs)]

    def download(self):
        pass

    def process(self):
        idx = 0
        image_ids_with_coco_captions_with_nodes = utilfunc_read_pickle_to_list("helper_files/image_ids_with_coco_captions_with_nodes.pickle")

        for image_id in tqdm(image_ids_with_coco_captions_with_nodes, total=len(image_ids_with_coco_captions_with_nodes)):
            #TODO: skip?
            if osp.exists(osp.join(self.processed_dir, f'data_{idx}.pt')):
                idx += 1
                continue
            file_path = f'graph_cleaned/data_{image_id}.pt'
            graph = None
            with open(file_path, 'rb') as fp:
                graph = torch.load(fp)
            G = nx.DiGraph()
            text_graph = nx.DiGraph()
            node_id = None
            node_ids = []
            node_classes = []
            node_features = []
            for node in graph.nodes:
                node_id = node
                node_ids.append(node_id)
                node_class = graph.nodes[node]['classes']
                node_classes.append(node_class)
            node_features = self._calculate_text_features_from_list_of_captions(node_classes)
            for index, node in enumerate(graph.nodes):
                G.add_node(node_ids[index], x = node_features[index])
                text_graph.add_node(node_ids[index], classes = node_classes[index])
            edge_classes = []
            edge_sentences = []
            edge_features = []
            for edge in graph.edges:
                 from_node, to_node = edge
                 from_node_class = graph.nodes[from_node]['classes']
                 edge_class = graph.edges[edge]['classes']
                 edge_classes.append(edge_class)
                 to_node_class = graph.nodes[to_node]['classes']
                 edge_sentence = from_node_class + " " + edge_class + " " + to_node_class
                 edge_sentences.append(edge_sentence)
            edge_features = self._calculate_text_features_from_list_of_captions(edge_sentences)
            for index, edge in enumerate(graph.edges):
                from_node, to_node = edge
                G.add_edge(from_node, to_node)
                text_graph.add_edge(from_node, to_node, classes = edge_classes[index])
            pyg_graph = from_networkx(G)
            pyg_graph.edge_attr = edge_features
            tuple = (pyg_graph, image_id, idx)

            torch.save(tuple, osp.join(self.processed_dir, f'data_{idx}.pt'))
            torch.save(text_graph, osp.join("text_graphs", f'data_{image_id}.pt'))
            idx += 1

    def _calculate_text_features_from_list_of_captions(self, list_of_captions):
        if len(list_of_captions) == 0:
            #list_of_captions = [""] #TODO Why did he used graphs without edges
            return torch.zeros(0, 256).to(device)
        text_tokens = blip_model.tokenizer(list_of_captions,
                    truncation=True,
                    padding=True,
                    max_length=blip_model.max_txt_len,
                    return_tensors="pt",
            ).to(device)

        with torch.no_grad():
            text_features = blip_model.Qformer.bert(
                text_tokens.input_ids,
                attention_mask=text_tokens.attention_mask,
                return_dict=True,
            )
            text_features = blip_model.text_proj(text_features.last_hidden_state[:, 0, :])
        
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        pyg_graph, image_idx, data_idx = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'), map_location=device)
        text_graph = torch.load(osp.join("text_graphs", f'data_{image_idx}.pt'), map_location=device)
        tuple = (pyg_graph, text_graph, image_idx, data_idx)
        return tuple

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'

#TODO: no se que es esto. De momento lo saco.
dataset = GraphDatasetBasic(root="graph_dataset_basic")