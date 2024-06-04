import random
random.seed(42)
import torch
torch.manual_seed(12345)
import torch.nn.functional as F
import numpy as np
from util import *

import blip_utils

from torch_geometric.nn import GATv2Conv, BatchNorm, Linear, AttentionalAggregation
from torch.nn import Linear

torch.backends.cudnn.deterministic = True  #< To anyone reading this: This is important. Settings seeds are not enough when using dropout. 

class AggregationModuleTanh(torch.nn.Module):
    '''
    This is the attention module. Its goal is to
    estimate the relative importance of each node for each
    features.
    '''
    def __init__(self, hidden_channels):
        super(AggregationModuleTanh, self).__init__()
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        pass

    def forward(self, x):
        '''
        Computes the relative importance of each node per each feature.
        I used a simple archetecture MLP architecture, as 
        in the original paper https://arxiv.org/pdf/1904.12787.pdf .
        Input shape nodes x features
        Output shape nodes x features
        '''
        p = self.lin1(x)
        p = F.tanh(p)
        return self.lin2(p)

class Encoder(torch.nn.Module):
    def __init__(self, hidden_channels, dropout_p):
        super(Encoder, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATv2Conv(in_channels=hidden_channels, out_channels=hidden_channels, edge_dim=hidden_channels)
        self.batchnorm1 = BatchNorm(hidden_channels)
        self.drop_layer1 = torch.nn.Dropout(p=dropout_p)

        self.conv2 = GATv2Conv(in_channels=hidden_channels, out_channels=hidden_channels, edge_dim=hidden_channels)
        self.batchnorm2 = BatchNorm(hidden_channels)
        self.drop_layer2 = torch.nn.Dropout(p=dropout_p)

        self.aggregation = AttentionalAggregation(AggregationModuleTanh(hidden_channels))
        self.lin = Linear(hidden_channels, hidden_channels)

        self.lin_image_embedder = Linear(hidden_channels, hidden_channels)
        self.batchnorm_image_embedder = BatchNorm(hidden_channels)
        self.drop_layer_image_embedder = torch.nn.Dropout(p=dropout_p)

        self.lin_text_embedder = Linear(hidden_channels, hidden_channels)
        self.batchnorm_text_embedder = BatchNorm(hidden_channels)
        self.drop_layer_text_embedder = torch.nn.Dropout(p=dropout_p)
    
    def forward(self, x, edge_index, edge_attr, batch, blip_image_features, blip_text_features):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.batchnorm1(x)
        x = self.drop_layer1(x)
        x = F.tanh(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.batchnorm2(x)
        x = self.drop_layer2(x)
        x = F.tanh(x)

        b = []
        for i in range(int(batch.max().item() + 1)):
            x_b = x[batch == i, :]
            b.append(self.aggregation(x_b))
        x = torch.vstack(b)
        x = self.lin(x)
        x = F.normalize(x, p=2, dim=-1)

        #First FC layer - images
        blip_image_features = self.lin_image_embedder(blip_image_features)
        blip_image_features = self.batchnorm_image_embedder(blip_image_features)
        blip_image_features = self.drop_layer_image_embedder(blip_image_features)
        blip_image_features = F.tanh(blip_image_features)
        blip_image_features = F.normalize(blip_image_features, p=2, dim=-1)
        
        #First FC layer - texts
        blip_text_features = self.lin_text_embedder(blip_text_features)
        blip_text_features = self.batchnorm_text_embedder(blip_text_features)
        blip_text_features = self.drop_layer_text_embedder(blip_text_features)
        blip_text_features = F.tanh(blip_text_features)
        blip_text_features = F.normalize(blip_text_features, p=2, dim=-1)

        return x, blip_image_features, blip_text_features

class TIGR_Blip2_att(torch.nn.Module):
    def __init__(self, hidden_channels, dropout_p, path_to_pretrained_blip_model = None,  path_to_pretrained_graph_encoder = None):
        super(TIGR_Blip2_att, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if path_to_pretrained_blip_model == None:
            self.blip_model, _ = blip_utils.load_default_blip2_model(device=device)
            self.blip_model.to(device)
        else:
            self.blip_model = torch.load(path_to_pretrained_blip_model)
            print("Pretrained blip loaded.")
            self.blip_model.to(device)
        
        if path_to_pretrained_graph_encoder == None:
            self.encoder = Encoder(hidden_channels=hidden_channels, dropout_p=dropout_p)
        else:
            self.encoder = torch.load(path_to_pretrained_graph_encoder)
            print("Pretrained encoder loaded.")
            self.encoder.to(device)

        self.att_1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.drop_layer1 = torch.nn.Dropout(p=dropout_p)
        self.att_2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.logit_scale_init_value = np.log(1 / 0.07)
        self.logit_scale_image_text = torch.nn.Parameter(torch.ones([]) * self.logit_scale_init_value)
        self.logit_scale_graph_image = torch.nn.Parameter(torch.ones([]) * self.logit_scale_init_value)
        self.logit_scale_graph_text = torch.nn.Parameter(torch.ones([]) * self.logit_scale_init_value)
        
    def forward(self, x, edge_index, edge_attr, batch, image, text):
        image_features = None
        text_features = None
        graph_features = None

        image_features = self.encode_image_with_foundation_model(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = self.encode_text_with_foundation_model(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        #Selecting best embedding according to BLIP2 technique 
        att = self.att_1(image_features)
        att = torch.nn.functional.relu(self.drop_layer1(att))
        att = self.att_2(att)
        att = torch.softmax(att, dim=1)
        image_features = torch.sum(att * image_features, dim=1)
        # Option 2
        # sims = torch.bmm(image_features, text_features.unsqueeze(-1))
        # sim, _idxs = torch.max(sims, dim=1)
        # image_features = image_features[torch.arange(128), _idxs.squeeze(1), :]
        # del sims
        # del sim
        # del _idxs
        #End patch for blip2
        graph_features, image_features, text_features = self.encoder(x, edge_index, edge_attr, batch, image_features, text_features)

        logit_scale_image_text = torch.clamp(self.logit_scale_image_text.exp(), min=1.0, max=100.0)
        logit_scale_graph_image = torch.clamp(self.logit_scale_graph_image.exp(), min=1.0, max=100.0)
        logit_scale_graph_text = torch.clamp(self.logit_scale_graph_text.exp(), min=1.0, max=100.0)

        logits_image_text = None
        logits_graph_image = None
        logits_graph_text = None

        logits_image_text = torch.matmul(image_features, text_features.t()) * logit_scale_image_text
        logits_graph_image = torch.matmul(graph_features, image_features.t()) * logit_scale_graph_image
        logits_graph_text = torch.matmul(graph_features, text_features.t()) * logit_scale_graph_text

        loss = self.loss_function(logits_image_text, logits_graph_image, logits_graph_text)

        return image_features, text_features, graph_features, loss

    def encode_image_with_foundation_model(self, image):
        # The image has already been preprocessed during training.
        with torch.cuda.amp.autocast(enabled=(self.blip_model.device != torch.device("cpu"))):
            image_embeds = self.blip_model.ln_vision(self.blip_model.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.blip_model.device)
        query_tokens = self.blip_model.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output =self. blip_model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_feats = F.normalize(
            self.blip_model.vision_proj(query_output.last_hidden_state), dim=-1
        )
        return image_feats

    def encode_text_with_foundation_model(self, text):
        text = self.blip_model.tokenizer(text,
                    truncation=True,
                    padding=True,
                    max_length=self.blip_model.max_txt_len,
                    return_tensors="pt",
            ).to(self.blip_model.device)
            
        text_output = self.blip_model.Qformer.bert(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(
            self.blip_model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )
        return text_feat
    
    def loss_function(self, logits_image_text, logits_graph_image, logits_graph_text):
        #First loss is calculated for image-text pairs.
        loss_image_text = self.contrastive_loss(logits_image_text)
        loss_text_image = self.contrastive_loss(logits_image_text.t())
        average_loss_image_text = (loss_image_text + loss_text_image) / 2.0

        #Then loss is calculated for graph-image pairs.
        loss_graph_image = self.contrastive_loss(logits_graph_image)
        loss_image_graph = self.contrastive_loss(logits_graph_image.t())
        average_loss_graph_image = (loss_graph_image + loss_image_graph) / 2.0

        #Then loss is calculated for graph-text pairs.
        loss_graph_text = self.contrastive_loss(logits_graph_text)
        loss_text_graph = self.contrastive_loss(logits_graph_text.t())
        average_loss_graph_text = (loss_graph_text + loss_text_graph) / 2.0

        return (average_loss_image_text + average_loss_graph_image + average_loss_graph_text) / 3.0
    
    def contrastive_loss(self, logits):
        return torch.nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))
    
    def check_if_any_weight_are_nan(self, model):
        is_nan = torch.stack([torch.isnan(p).any() for p in model.parameters()]).any()
        return is_nan
    
    def freeze_blip_model_weights(self):
        for param in self.blip_model.parameters():
            param.requires_grad = False

    def unfreeze_blip_model_weights(self):
        for param in self.blip_model.parameters():
            param.requires_grad = True