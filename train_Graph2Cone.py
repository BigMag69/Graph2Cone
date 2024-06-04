import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '7'
import random
random.seed(42)
from tqdm.auto import tqdm
import numpy as np
import json
import matplotlib.pyplot as plt
from util import *
from torch_geometric.loader import DataLoader
from models import Graph2Cone, Graph2Cone_hybrid, cone_loss, cone_loss_skewed, cone_loss2
#from basic_data import GraphDatasetBestBlip2CleanedSentencesOnEdgesPreprocessed
#from data import GraphDatasetWithNegativeSamples
#from data4 import GraphDatasetEdgesWithNegativeSamples
from data6 import GraphDatasetEdgesAndNodesWithNegativeSamples
import blip_utils
import torchvision.transforms as transforms

import torch
from torch_geometric import nn
torch.manual_seed(12345)

from IPython.display import display, clear_output

from torch.optim.lr_scheduler import ReduceLROnPlateau
from IPython.display import Javascript
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

print("Torch version:", torch.__version__)
print("Numpy version:", np.__version__)

device = "cuda:1" if torch.cuda.is_available() else "cpu"
print("Device: ", device)

#torch.autograd.set_detect_anomaly(True)
############################### Set up ############################### 

def get_config_dict(): # upto 5 edges # 1edge # 1 per graph
    config_dict = {}
    config_dict["training_set_end_index"] = 150312 #190663 #72491 #40783
    config_dict["validation_set_end_index"] = 167014 #200663 #77491 #45783
    config_dict["batch_size"] = 128
    config_dict["dropout_p"] = 0.1 
    config_dict["learning_rate_adam"] = 1e-4
    config_dict["weight_decay_adam"] =  1e-5
    config_dict["reduce_lr_on_plateau_factor"] = 0.1
    config_dict["reduce_lr_on_plateau_patience"] = 2 
    config_dict["reduce_lr_on_plateau_threshold"] = 0.001
    config_dict["hidden_channels_in_network"] = 256 #TODO is the projection layer.... or should not use it?
    config_dict["optimizer_string"] = "Adam"
    config_dict["scheduler_string"] = "ReduceLROnPlateau"
    config_dict["early_stopping"] = True
    config_dict["num_epochs"] = 200
    config_dict["threshold_multiplier_for_early_stopping"] = 0.001
    config_dict["early_stopping_patience"] = 5
    config_dict["positve_gamma"] = 12
    config_dict["negative_gamma"] = 12
    config_dict["positive_lambda"] = 0.2
    config_dict["negative_lambda"] = 0.2
    return config_dict

config_dict = get_config_dict()
#Hyper parameters (This could be cleaned up - I originally got the parameters from another file, but I am leaving as is)
training_set_end_index = config_dict['training_set_end_index']
validation_set_end_index = config_dict["validation_set_end_index"]
batch_size = config_dict["batch_size"]
dropout_p = config_dict["dropout_p"] 
learning_rate_adam = config_dict["learning_rate_adam"] 
weight_decay_adam = config_dict["weight_decay_adam"] 
reduce_lr_on_plateau_factor = config_dict["reduce_lr_on_plateau_factor"]
reduce_lr_on_plateau_patience = config_dict["reduce_lr_on_plateau_patience"]
reduce_lr_on_plateau_threshold = config_dict["reduce_lr_on_plateau_threshold"]
hidden_channels_in_network = config_dict["hidden_channels_in_network"]
optimizer_string = config_dict["optimizer_string"]
scheduler_string = config_dict["scheduler_string"]
num_epochs = config_dict["num_epochs"]
early_stopping = config_dict["early_stopping"]
threshold_multiplier_for_early_stopping = config_dict["threshold_multiplier_for_early_stopping"]
early_stopping_patience = config_dict["early_stopping_patience"]
positve_gamma = config_dict["positve_gamma"]
negative_gamma = config_dict["negative_gamma"]
positive_lamb = config_dict["positive_lambda"]
negative_lamb = config_dict["negative_lambda"]


device = "cuda:1" if torch.cuda.is_available() else "cpu"
_, preprocess = blip_utils.load_default_blip2_model(device=device)

image_ids_with_coco_captions_with_nodes = utilfunc_read_pickle_to_list("helper_files/image_ids_with_coco_captions_with_nodes.pickle")


model = Graph2Cone(hidden_channels=hidden_channels_in_network, dropout_p=dropout_p)
#load model from checkpoint
#model = torch.load("Graph2Cone_saved_files/Graph2Cone_full_model_2024-05-07_15-08-02.pt")
model.to(device)

dataset = GraphDatasetEdgesAndNodesWithNegativeSamples()

print("----------------------- Remember to check if everything can terminate - Check with a couple of epochs -----------------------------------")
print("----------------------- How you remembered to set string to save the model statistics? --------------------------------------------------")
print("----------------------- How you changed the model accordingly? --------------------------------------------------------------------------")

string_for_files = "Graph2Cone_finetuning_0"

train_dataset = dataset[:training_set_end_index]
val_dataset = dataset[training_set_end_index:validation_set_end_index]
test_dataset = dataset[validation_set_end_index:]

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


if optimizer_string == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_adam, weight_decay=weight_decay_adam)

# if optimizer_string == "AdamW":
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate_adamw, weight_decay=weight_decay_adamw)

# if optimizer_string == "SGD":
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate_sgd, momentum=0.9, nesterov=True, weight_decay=weight_decay_sgd)
if scheduler_string == "ReduceLROnPlateau":
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=reduce_lr_on_plateau_factor, patience=reduce_lr_on_plateau_patience, threshold=reduce_lr_on_plateau_threshold, verbose=True)


#####################################################################################################################################

#####################################################################################################

def config_saver(config_path, config_dict):
    with open(config_path, 'w') as file:
        dictionary_str = json.dumps(config_dict)
        file.write(dictionary_str)
        file.write('\n')

def plot_saver(loss_curve_path, train_losses, val_losses):
    plt.figure(figsize=(10,5))
    plt.title("Training/validation Loss")
    plt.plot(train_losses,label="train")
    plt.plot(val_losses,label="val")
    ax = plt.gca()
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(loss_curve_path, bbox_inches='tight')
    plt.close()

def loss_file_saver(loss_file_path, train_losses, val_losses):
    loss_string = "val_losses = " + str(val_losses) + "\n"
    loss_string += "train_losses = " + str(train_losses) + "\n"
    
    f = open(loss_file_path, "w")
    f.write(loss_string)
    f.close()

############################### Training ###############################

def eval():
    model.eval()
    v_loss = []
    with torch.no_grad():
        for q2, q1, negative_samples, image_idx, data_idx in tqdm(val_loader, total=len(val_loader)):
            q1 = q1.to(device)
            q2 = q2.to(device)
            negative_samples = [negative_sample.to(device) for negative_sample, _ in negative_samples]
            val_loss = cone_loss2(model(q1.x, q1.edge_index, q1.edge_attr, q1.batch), 
                                  model(q2.x, q2.edge_index, q2.edge_attr, q2.batch), 
                                  [model(neg.x, neg.edge_index, neg.edge_attr, neg.batch) for neg in negative_samples], 
                                  postive_lamb=positive_lamb, 
                                  negative_lamb=negative_lamb, 
                                  positive_gamma=positve_gamma, 
                                  negative_gamma=negative_gamma)
            val_loss = val_loss.cuda().mean()
            v_loss.append(val_loss.item())
    return np.mean(v_loss)

def train(index_for_captions):
    index_for_captions_for_training = index_for_captions
    model.train()
    t_loss = []
    for q2, q1, negative_samples, image_idx, data_idx in tqdm(train_loader, total=len(train_loader)):
        optimizer.zero_grad()
        q1 = q1.to(device)
        q2 = q2.to(device)
        negative_samples = [negative_sample.to(device) for negative_sample, _ in negative_samples]
        train_loss = cone_loss2(model(q1.x, q1.edge_index, q1.edge_attr, q1.batch), 
                                model(q2.x, q2.edge_index, q2.edge_attr, q2.batch), 
                                [model(neg.x, neg.edge_index, neg.edge_attr, neg.batch) for neg in negative_samples], 
                                postive_lamb=positive_lamb, 
                                negative_lamb=negative_lamb, 
                                positive_gamma=positve_gamma, 
                                negative_gamma=negative_gamma)
        train_loss = train_loss.cuda().mean()
        train_loss.backward()
        optimizer.step()
        t_loss.append(train_loss.item())
        index_for_captions_for_training = index_for_captions_for_training + 1
        index_for_captions_for_training = index_for_captions_for_training % 5
    val_loss = eval()
    scheduler.step(val_loss)

    return np.mean(t_loss), val_loss, index_for_captions_for_training


if __name__=="__main__":
    timestamp = utilfunc_get_timestamp_string()
    if not os.path.exists('Graph2Cone_saved_files'):
        os.makedirs('Graph2Cone_saved_files')
    full_model_path = "Graph2Cone_saved_files/" + string_for_files + "_full_model_" + timestamp + ".pt"
    encoder_path = "Graph2Cone_saved_files/" + string_for_files + "_encoder_" + timestamp + ".pt"
    statistics_file_path = "Graph2Cone_saved_files/" + string_for_files + "_statistics_" + timestamp + ".txt"
    aperture_file_path = "Graph2Cone_saved_files/" + string_for_files + "_aperture_" + timestamp + ".txt"
    loss_file_path = "Graph2Cone_saved_files/" + string_for_files + "_loss_file_" + timestamp + ".txt"
    loss_curve_path = "Graph2Cone_saved_files/" + string_for_files + "_loss_plot_" + timestamp + ".png"
    config_dict_file_path = "Graph2Cone_saved_files/" + string_for_files + "_config_dict_" + timestamp + ".txt"
    print("Model path:" + full_model_path)
    print("Statistics path:" + statistics_file_path)
    print("Is that right?")
    train_losses = []
    val_losses = []
    initial_val_loss = eval()
    print("Initial validation loss: ", initial_val_loss)

    #For early stopping
    best_val_loss = initial_val_loss
    counter = 0
    index_for_captions = 0
    for epoch in tqdm(range(1, num_epochs+1), total=num_epochs):
        train_loss, val_loss, index_for_captions_new = train(index_for_captions)
        index_for_captions = index_for_captions_new
        print(f'Epoch: {epoch:03d}, Loss, train: {train_loss}, Loss, val: {val_loss}, - For debugging: Counter for early stopping: {counter}, Best loss, val: {best_val_loss}')
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if early_stopping: 
            if (val_loss < best_val_loss) and (best_val_loss - val_loss) > (best_val_loss * threshold_multiplier_for_early_stopping):
                torch.save(model, full_model_path)
                plot_saver(loss_curve_path, train_losses, val_losses)
                loss_file_saver(loss_file_path, train_losses, val_losses)
                config_saver(config_dict_file_path, config_dict)
                best_val_loss = val_loss
                counter = 0
            else:
                print("Best val. loss remains " + str(best_val_loss))
                counter += 1
        
            if counter >= early_stopping_patience:
                print("Stopping early: Validation loss has not improved for " + str(early_stopping_patience) + "epochs.")
                print("Breaking early after " + str(epoch) + " epochs.")
                break
        else:
            if val_loss < best_val_loss:
                torch.save(model, full_model_path)
                plot_saver(loss_curve_path, train_losses, val_losses)
                loss_file_saver(loss_file_path, train_losses, val_losses)
                config_saver(config_dict_file_path, config_dict)
                best_val_loss = val_loss
    
    if val_loss < best_val_loss:
        torch.save(model, full_model_path)
        plot_saver(loss_curve_path, train_losses, val_losses)
        loss_file_saver(loss_file_path, train_losses, val_losses)
        config_saver(config_dict_file_path, config_dict)

    plt.figure(figsize=(10,5))
    plt.title("Training/validation Loss")
    plt.plot(train_losses,label="train")
    plt.plot(val_losses,label="val")
    ax = plt.gca()
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(loss_curve_path, bbox_inches='tight')
    plt.close()
    
    loss_string = "val_losses = " + str(val_losses) + "\n"
    loss_string += "train_losses = " + str(train_losses) + "\n"
    
    f = open(loss_file_path, "a")
    f.write(loss_string)
    f.close()

    print("Training has concluded.")