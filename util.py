
from typing import List, Optional

import clip
import random
random.seed(42)
from tqdm.auto import tqdm
import numpy as np
import networkx as nx

import PIL

import torch
torch.manual_seed(12345)
import pickle
import datetime

# ------------------------------------------------------------------------- Graph functionality ------------------------------------------------------------------------- #

def utilfunc_load_cleaned_graphs_as_dict(image_ids_with_coco_captions_with_nodes):
    graph_dict = {}
    for image_id in tqdm(image_ids_with_coco_captions_with_nodes, total=len(image_ids_with_coco_captions_with_nodes)):
        file_path = "graph_pickles_cleaned/" + str(image_id) + ".pickle"
        graph = None
        with open(file_path, 'rb') as fp:
            graph = pickle.load(fp)
        graph_dict[image_id] = graph
    return graph_dict

# ------------------------------------------------------------------------- Caption functionality ------------------------------------------------------------------------- #

# def utilfunc_coco_captions_without_empty_captions():
#     coco_captions = get_coco_caption('data')
#     coco_captions_without_empty_captions = {}
#     for k, v in list(coco_captions.items()):
#         if v != []:
#             coco_captions_without_empty_captions[k] = v
#     del coco_captions
#     return coco_captions_without_empty_captions

def utilfunc_image_ids_with_coco_captions_with_nodes(g, keys):
    image_ids_with_coco_captions = sorted(list(keys))
    image_ids_with_coco_captions_but_without_nodes = []
    for image_id in image_ids_with_coco_captions:
        if len(g[image_id].nodes) == 0:
            image_ids_with_coco_captions_but_without_nodes.append(image_id)

    image_ids_with_coco_captions_with_nodes = sorted(list(set(image_ids_with_coco_captions) - set(image_ids_with_coco_captions_but_without_nodes)))
    return image_ids_with_coco_captions_with_nodes

# ------------------------------------------------------------------------- Pickle functionality ------------------------------------------------------------------------- #

def utilfunc_write_list_to_pickle(list, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(list, f)
    print("Wrote list to " + str(file_path))

def utilfunc_read_pickle_to_list(file_path):
    list_from_pickle = None
    with open(file_path, 'rb') as fp:
        list_from_pickle = pickle.load(fp)
    return list_from_pickle

def utilfunc_read_image_ids_with_coco_captions_with_nodes_from_pickle(filename):
    image_ids_with_coco_captions_with_nodes = None
    with open(filename, 'rb') as fp:
        image_ids_with_coco_captions_with_nodes = pickle.load(fp)
    return image_ids_with_coco_captions_with_nodes

# def utilfunc_original_graphs_to_pickle():
#     g = load_full_vg_14('data')
#     coco_captions_without_empty_captions = utilfunc_coco_captions_without_empty_captions()
#     image_ids_with_coco_captions_with_nodes = utilfunc_image_ids_with_coco_captions_with_nodes(g, coco_captions_without_empty_captions)

#     for image_id in tqdm(image_ids_with_coco_captions_with_nodes, total=len(image_ids_with_coco_captions_with_nodes)):
#         folder = "graph_pickles/"
#         file_path = folder + str(image_id) + ".pickle"

#         with open(file_path, 'wb') as f:
#             G = nx.DiGraph()
#             for node in g[image_id].nodes:
#                 node_id = node
#                 node_class = g[image_id].nodes[node].classes[0]
#                 G.add_node(node_id, node_class = node_class)
#             for edge in g[image_id].edges:
#                 from_node, to_node = edge
#                 edge_class = g[image_id].edges[edge].classes[0]
#                 G.add_edge(from_node, to_node, edge_class = edge_class)
#             pickle.dump(G, f)

def utilfunc_cleaned_graphs_to_pickle(graph, image_ids_with_coco_captions_with_nodes):
    for image_id in tqdm(image_ids_with_coco_captions_with_nodes, total=len(image_ids_with_coco_captions_with_nodes)):
        folder = "graph_pickles_cleaned/"
        file_path = folder + str(image_id) + ".pickle"

        with open(file_path, 'wb') as f:
            pickle.dump(graph[image_id], f)

# ------------------------------------------------------------------------- Embeddings functionality ------------------------------------------------------------------------- #

# Lots of this is old stuff. 

def utilfunc_load_image_embeddings_from_files():
    all_image_embeddings = []
    prefix_for_file_path_for_embeddings = "image_embeddings/image_features_part"
    file_numbers = list(range(1,52)) 
    file_paths = [prefix_for_file_path_for_embeddings + str(file_number) + ".pt" for file_number in file_numbers]
    for file_index, file_path in enumerate(file_paths):
        image_features_from_file = torch.load(file_path)
        for image_feature in image_features_from_file:
                all_image_embeddings.append(image_feature)
    return all_image_embeddings

def utilfunc_load_image_embeddings_better_clip_from_files():
    all_image_embeddings = []
    prefix_for_file_path_for_embeddings = "image_embeddings_better_clip/image_features_part"
    file_numbers = list(range(1,509)) 
    file_paths = [prefix_for_file_path_for_embeddings + str(file_number) + ".pt" for file_number in file_numbers]
    for file_index, file_path in enumerate(file_paths):
        image_features_from_file = torch.load(file_path)
        for image_feature in image_features_from_file:
                all_image_embeddings.append(image_feature)
    return all_image_embeddings

def utilfunc_load_graphs_embeddings_from_test_set_from_files():
    graph_embeddings = []
    prefix_for_file_path_for_embeddings = "graph_embeddings/graph_embeddings_part"
    file_numbers = list(range(1,8)) 
    file_paths = [prefix_for_file_path_for_embeddings + str(file_number) + ".pt" for file_number in file_numbers]
    for file_index, file_path in enumerate(file_paths):
        graph_features_from_file = torch.load(file_path)
        for graph_feature in graph_features_from_file:
                graph_embeddings.append(graph_feature)
    return graph_embeddings

def utilfunc_load_text_embeddings_from_files():
    file_path = "text_embeddings/text_embeddings.pt"
    text_embeddings = torch.load(file_path)
    return text_embeddings

def utilfunc_load_images_embeddings_from_file(file_path):
    image_embeddings = torch.load(file_path)
    return image_embeddings

def utilfunc_calculate_text_embeddings_from_list_of_captions(clip_model, list_of_captions):
        text_tokens = clip.tokenize(list_of_captions).cuda()

        with torch.no_grad():
            text_embeddings = clip_model.encode_text(text_tokens).float()
        
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings

def utilfunc_calculate_image_embeddings_from_list_of_image_ids(clip_model, preprocess, array_of_image_ids):
    processed_images = []
    for image_id in array_of_image_ids:
        image = utilfunc_image_opener_without_zip(image_id)
        processed_images.append(preprocess(image))
    
    image_input = torch.tensor(np.stack(processed_images)).cuda()
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input).float()
    
    image_features /= image_features.norm(dim=-1, keepdim=True)

    return image_features

# ------------------------------------------------------------------------- Image functionality ------------------------------------------------------------------------- #

def utilfunc_image_opener_without_zip(image_id: int):
    file_path = "data/VG1.4/images/VG_100K/" + str(image_id) + ".jpg"
    try:
        image = PIL.Image.open(file_path)
    except FileNotFoundError:
        file_path = "data/VG1.4/images2/VG_100K_2/" + str(image_id) + ".jpg"
        image = PIL.Image.open(file_path)
    return image

def utilfunc_image_opener_with_list_without_zip(image_ids):
    images = []
    for image_id in image_ids.tolist():
        file_path = "data/VG1.4/images/VG_100K/" + str(image_id) + ".jpg"
        try:
            image = PIL.Image.open(file_path)
        except FileNotFoundError:
            file_path = "data/VG1.4/images2/VG_100K_2/" + str(image_id) + ".jpg"
            image = PIL.Image.open(file_path)
        images.append(image)
    return images

# ------------------------------------------------------------------------- Plotting functionality ------------------------------------------------------------------------- #

def utilfunc_get_timestamp_string():
    now = datetime.datetime.now()
    timestamp = str(now).split(".")[0]
    timestamp = timestamp.replace(":", "-")
    timestamp = timestamp.replace(" ", "_")
    return timestamp

# ------------------------------------------------------------------------- Training functionality ------------------------------------------------------------------------- #

#This is not used anymore, but I am leaving it for completeness. 
#In the beginning I found it easier to have all parameters in this file.
#I since split it out on multiple files. 
def utilfunc_get_config_dict():
    config_dict = {}
    config_dict["training_set_end_index"] = 40783 
    config_dict["validation_set_end_index"] = 45783 
    config_dict["dropout_p"] = 0.1 
    config_dict["hidden_channels_in_network"] = 768
    config_dict["clip_model_string"] = 'ViT-L/14@336px'
    return config_dict

#Clip models:
# ['RN50',
#  'RN101',
#  'RN50x4',
#  'RN50x16',
#  'RN50x64',
#  'ViT-B/32',
#  'ViT-B/16',
#  'ViT-L/14',
#  'ViT-L/14@336px']