import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm, Linear, AttentionalAggregation
from torch.nn import Linear

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

class Graph2Cone(torch.nn.Module):
    def __init__(self, hidden_channels, dropout_p):
        super(Graph2Cone, self).__init__()
        self.hidden_channels = hidden_channels

        self.conv1 = GATv2Conv(in_channels=hidden_channels, out_channels=hidden_channels*2, edge_dim=hidden_channels)
        self.batchnorm1 = BatchNorm(hidden_channels*2)
        self.drop_layer1 = torch.nn.Dropout(p=dropout_p)

        self.conv2 = GATv2Conv(in_channels=hidden_channels*2, out_channels=hidden_channels*2, edge_dim=hidden_channels)
        self.batchnorm2 = BatchNorm(hidden_channels*2)
        self.drop_layer2 = torch.nn.Dropout(p=dropout_p)

        self.aggregation1 = AttentionalAggregation(AggregationModuleTanh(hidden_channels*2))
        self.lin1 = Linear(hidden_channels*2, hidden_channels*2)
        #self.lin1.bias.data.fill_(-1)

        # Constants for scaling
        self.pi = torch.tensor(3.14159265358979323846)

    def forward(self, x, edge_index, edge_attr, batch, is_query=True):
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
            b.append(self.aggregation1(x_b))
        x = torch.vstack(b)
        x = self.lin1(x)
        x = F.tanh(x)

        # Reshape and scale outputs for axis and aperture
        axis, aperture = torch.chunk(x, 2, dim=-1)
        axis = axis * self.pi  # Scale to [-pi, pi] # F.tanh(axis) * 
        aperture = (aperture + 1) * self.pi # Scale to [0, 2pi] # F.sigmoid(aperture) * 2 * self.pi
        if not is_query:
            aperture = torch.zeros_like(aperture)

        return axis, aperture

class Graph2Cone128(torch.nn.Module):
    def __init__(self, hidden_channels, dropout_p):
        super(Graph2Cone128, self).__init__()
        self.hidden_channels = hidden_channels

        self.conv1 = GATv2Conv(in_channels=hidden_channels, out_channels=hidden_channels, edge_dim=hidden_channels)
        self.batchnorm1 = BatchNorm(hidden_channels)
        self.drop_layer1 = torch.nn.Dropout(p=dropout_p)

        self.conv2 = GATv2Conv(in_channels=hidden_channels, out_channels=hidden_channels, edge_dim=hidden_channels)
        self.batchnorm2 = BatchNorm(hidden_channels)
        self.drop_layer2 = torch.nn.Dropout(p=dropout_p)

        self.aggregation1 = AttentionalAggregation(AggregationModuleTanh(hidden_channels))
        self.lin1 = Linear(hidden_channels, hidden_channels)
        #self.lin1.bias.data.fill_(-1)

        # Constants for scaling
        self.pi = torch.tensor(3.14159265358979323846)

    def forward(self, x, edge_index, edge_attr, batch, is_query=True):
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
            b.append(self.aggregation1(x_b))
        x = torch.vstack(b)
        x = self.lin1(x)
        x = F.tanh(x)

        # Reshape and scale outputs for axis and aperture
        axis, aperture = torch.chunk(x, 2, dim=-1)
        axis = axis * self.pi  # Scale to [-pi, pi] # F.tanh(axis) * 
        aperture = (aperture + 1) * self.pi # Scale to [0, 2pi] # F.sigmoid(aperture) * 2 * self.pi
        if not is_query:
            aperture = torch.zeros_like(aperture)

        return axis, aperture
    
class Graph2Cone3Layers(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(Graph2Cone3Layers, self).__init__()
        self.hidden_channels = hidden_channels

        self.lin0 = Linear(hidden_channels, hidden_channels*2)
        self.lin1 = Linear(hidden_channels*2, hidden_channels*2)
        self.lin2 = Linear(hidden_channels*2, hidden_channels*2)

        # Constants for scaling
        self.pi = torch.tensor(3.14159265358979323846)

    def forward(self, x, edge_index, edge_attr, batch, is_query=True):
        x = self.lin0(x)
        x = F.relu(x)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)


        # Reshape and scale outputs for axis and aperture
        axis, aperture = torch.chunk(x, 2, dim=-1)
        axis = F.tanh(axis) * self.pi  # Scale to [-pi, pi] # F.tanh(axis) * 
        aperture = (F.tanh(aperture) + 1) * self.pi # Scale to [0, 2pi] # F.sigmoid(aperture) * 2 * self.pi
        return axis, aperture

class Answer2Cone(torch.nn.Module):
    def __init__(self, hidden_channels, dropout_p):
        super(Answer2Cone, self).__init__()
        self.hidden_channels = hidden_channels

        self.conv1 = GATv2Conv(in_channels=hidden_channels, out_channels=hidden_channels, edge_dim=hidden_channels)
        self.batchnorm1 = BatchNorm(hidden_channels)
        self.drop_layer1 = torch.nn.Dropout(p=dropout_p)

        self.conv2 = GATv2Conv(in_channels=hidden_channels, out_channels=hidden_channels, edge_dim=hidden_channels)
        self.batchnorm2 = BatchNorm(hidden_channels)
        self.drop_layer2 = torch.nn.Dropout(p=dropout_p)

        self.aggregation1 = AttentionalAggregation(AggregationModuleTanh(hidden_channels))
        self.lin1 = Linear(hidden_channels, hidden_channels)
        #self.lin1.bias.data.fill_(-1)

        # Constants for scaling
        self.pi = torch.tensor(3.14159265358979323846)

    def forward(self, x, edge_index, edge_attr, batch):
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
            b.append(self.aggregation1(x_b))
        x = torch.vstack(b)
        x = self.lin1(x)
        x = F.tanh(x)

        axis = x * self.pi  # Scale to [-pi, pi] # F.tanh(axis) * 
        aperture = torch.zeros_like(axis)
        return axis, aperture

class AggregationModuleRelu(torch.nn.Module):
    '''
    This is the attention module. Its goal is to
    estimate the relative importance of each node for each
    features.
    '''
    def __init__(self, hidden_channels):
        super(AggregationModuleRelu, self).__init__()
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

    def forward(self, x):
        '''
        Computes the relative importance of each node per each feature.
        I used a simple archetecture MLP architecture, as 
        in the original paper https://arxiv.org/pdf/1904.12787.pdf .
        Input shape nodes x features
        Output shape nodes x features
        '''
        p = self.lin1(x)
        p = F.relu(p)
        return self.lin2(p)

class Graph2Cone_relu(torch.nn.Module):
    def __init__(self, hidden_channels, dropout_p):
        super(Graph2Cone_relu, self).__init__()
        self.hidden_channels = hidden_channels

        self.conv1 = GATv2Conv(in_channels=hidden_channels, out_channels=hidden_channels*2, edge_dim=hidden_channels)
        self.batchnorm1 = BatchNorm(hidden_channels*2)
        self.drop_layer1 = torch.nn.Dropout(p=dropout_p)

        self.conv2 = GATv2Conv(in_channels=hidden_channels*2, out_channels=hidden_channels*2, edge_dim=hidden_channels)
        self.batchnorm2 = BatchNorm(hidden_channels*2)
        self.drop_layer2 = torch.nn.Dropout(p=dropout_p)

        self.aggregation1 = AttentionalAggregation(AggregationModuleRelu(hidden_channels*2))
        self.lin1 = Linear(hidden_channels*2, hidden_channels*2)
        #self.lin1.bias.data.fill_(-1)

        # Constants for scaling
        self.pi = torch.tensor(3.14159265358979323846)

    def forward(self, x, edge_index, edge_attr, batch, is_query=True):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.batchnorm1(x)
        x = self.drop_layer1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.batchnorm2(x)
        x = self.drop_layer2(x)
        x = F.relu(x)

        b = []
        for i in range(int(batch.max().item() + 1)):
            x_b = x[batch == i, :]
            b.append(self.aggregation1(x_b))
        x = torch.vstack(b)
        x = self.lin1(x)
        x = F.tanh(x)

        # Reshape and scale outputs for axis and aperture
        axis, aperture = torch.chunk(x, 2, dim=-1)
        axis = axis * self.pi  # Scale to [-pi, pi] # F.tanh(axis) * 
        aperture = (aperture + 1) * self.pi # Scale to [0, 2pi] # F.sigmoid(aperture) * 2 * self.pi
        if not is_query:
            aperture = torch.zeros_like(aperture)

        return axis, aperture

class Answer2Cone_relu(torch.nn.Module):
    def __init__(self, hidden_channels, dropout_p):
        super(Answer2Cone, self).__init__()
        self.hidden_channels = hidden_channels

        self.conv1 = GATv2Conv(in_channels=hidden_channels, out_channels=hidden_channels, edge_dim=hidden_channels)
        self.batchnorm1 = BatchNorm(hidden_channels)
        self.drop_layer1 = torch.nn.Dropout(p=dropout_p)

        self.conv2 = GATv2Conv(in_channels=hidden_channels, out_channels=hidden_channels, edge_dim=hidden_channels)
        self.batchnorm2 = BatchNorm(hidden_channels)
        self.drop_layer2 = torch.nn.Dropout(p=dropout_p)

        self.aggregation1 = AttentionalAggregation(AggregationModuleRelu(hidden_channels))
        self.lin1 = Linear(hidden_channels, hidden_channels)
        #self.lin1.bias.data.fill_(-1)

        # Constants for scaling
        self.pi = torch.tensor(3.14159265358979323846)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.batchnorm1(x)
        x = self.drop_layer1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.batchnorm2(x)
        x = self.drop_layer2(x)
        x = F.relu(x)

        b = []
        for i in range(int(batch.max().item() + 1)):
            x_b = x[batch == i, :]
            b.append(self.aggregation1(x_b))
        x = torch.vstack(b)
        x = self.lin1(x)
        x = F.tanh(x)

        axis = x * self.pi  # Scale to [-pi, pi] # F.tanh(axis) * 
        aperture = torch.zeros_like(axis)
        return axis, aperture


class Graph2Cone_hybrid(torch.nn.Module):
    def __init__(self, hidden_channels, dropout_p):
        super(Graph2Cone_hybrid, self).__init__()
        self.hidden_channels = hidden_channels

        self.conv1 = GATv2Conv(in_channels=hidden_channels, out_channels=hidden_channels*2, edge_dim=hidden_channels)
        self.batchnorm1 = BatchNorm(hidden_channels*2)
        self.drop_layer1 = torch.nn.Dropout(p=dropout_p)

        self.conv2 = GATv2Conv(in_channels=hidden_channels*2, out_channels=hidden_channels*2, edge_dim=hidden_channels)
        self.batchnorm2 = BatchNorm(hidden_channels*2)
        self.drop_layer2 = torch.nn.Dropout(p=dropout_p)

        self.aggregation1 = AttentionalAggregation(AggregationModuleTanh(hidden_channels*2))
        self.lin1 = Linear(hidden_channels*2, hidden_channels*2)
        #self.lin1.bias.data.fill_(-1)

        self.axis_layer = Linear(hidden_channels*2, hidden_channels)
        self.aperture_layer = Linear(hidden_channels*2, hidden_channels)

        # Constants for scaling
        self.pi = torch.tensor(3.14159265358979323846)

    def forward(self, x, edge_index, edge_attr, batch):
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
            b.append(self.aggregation1(x_b))
        x = torch.vstack(b)
        x = self.lin1(x)
        x = F.tanh(x)

        axis = self.axis_layer(x)
        aperture = self.aperture_layer(x)

        # Reshape and scale outputs for axis and aperture
        #axis, aperture = torch.chunk(x, 2, dim=-1)
        axis = axis * self.pi  # Scale to [-pi, pi] # F.tanh(axis) * 
        aperture = (aperture + 1) * self.pi  # Scale to [0, 2pi] # F.sigmoid(aperture) * 2 * self.pi

        return axis, aperture

# loss function
def sine_halves(theta):
    return torch.sin(theta / 2)

def d_con3(V_q1, V_q2, lamb=0):
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

    distance_out = torch.norm(distance_U_out, p=1, dim=-1) + torch.norm(distance_L_out, p=1, dim=-1) + torch.norm(distance_axis_out, p=1, dim=-1)

    distance_U_in = torch.norm(torch.min(distanceU2axis, distance_base), p=1, dim=-1)
    distance_L_in = torch.norm(torch.min(distanceL2axis, distance_base), p=1, dim=-1)
    distance_axis2axis = torch.norm(torch.min(distance_base, distanceax2axis), p=1, dim=-1)

    distance_in = distance_U_in + distance_L_in + distance_axis2axis

    distance = distance_out + lamb*distance_in
    return distance

def d_con4(V_q1, V_q2, lamb=0):
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

    distance_out = torch.norm(distance_axis_out, p=1, dim=-1)

    distance_axis2axis = torch.norm(torch.min(distance_base, distanceax2axis), p=1, dim=-1)
    distance_in = distance_axis2axis

    distance = distance_out + lamb*distance_in
    return distance

def cone_loss(q1_pos, q2, negative_sample_embeddings, lamb, gamma=36.0):
    k = len(negative_sample_embeddings)
    positive_loss = -F.logsigmoid(gamma - d_con3(q1_pos, q2, lamb))
    negative_loss = 0
    for q1_neg in negative_sample_embeddings:
        negative_loss += torch.sum(F.logsigmoid(d_con3(q1_neg, q2, lamb) - gamma))
    return positive_loss - 1/k * negative_loss

def cone_loss_no_avg(q1_pos, q2, negative_sample_embeddings, postive_lamb, negative_lamb, positive_gamma, negative_gamma):
    k = len(negative_sample_embeddings)
    positive_loss = -F.logsigmoid(positive_gamma - d_con3(q1_pos, q2, postive_lamb))
    negative_loss = 0
    for q1_neg in negative_sample_embeddings:
        negative_loss += torch.sum(F.logsigmoid(d_con3(q1_neg, q2, negative_lamb) - negative_gamma))
    return positive_loss - negative_loss

def cone_loss2(q1_pos, q2, negative_sample_embeddings, postive_lamb, negative_lamb, positive_gamma, negative_gamma):
    k = len(negative_sample_embeddings)
    positive_loss = -F.logsigmoid(positive_gamma - d_con3(q1_pos, q2, postive_lamb))
    negative_loss = 0
    for q1_neg in negative_sample_embeddings:
        negative_loss += torch.sum(F.logsigmoid(d_con3(q1_neg, q2, negative_lamb) - negative_gamma))
    return positive_loss - 1/k * negative_loss

def cone_loss3(q1_pos, q2, negative_sample_embeddings, postive_lamb, negative_lamb, positive_gamma, negative_gamma):
    k = len(negative_sample_embeddings)
    positive_loss = -F.logsigmoid(positive_gamma - d_con4(q1_pos, q2, postive_lamb))
    negative_loss = 0
    for q1_neg in negative_sample_embeddings:
        negative_loss += torch.sum(F.logsigmoid(d_con4(q1_neg, q2, negative_lamb) - negative_gamma))
    return positive_loss - 1/k * negative_loss

def cone_loss_union(q1_pos, q2s, negative_sample_embeddings, postive_lamb, negative_lamb, positive_gamma, negative_gamma):
    k = len(negative_sample_embeddings)
    positive_loss = -F.logsigmoid(positive_gamma - torch.min(d_con4(q1_pos, q2s, postive_lamb)))
    negative_loss = 0
    for q1_neg in negative_sample_embeddings:
        negative_loss += torch.sum(F.logsigmoid(torch.min(d_con4(q1_neg, q2s, negative_lamb)) - negative_gamma))
    return positive_loss - 1/k * negative_loss

def cone_loss_intersection(q1_pos, q2s, negative_sample_embeddings, postive_lamb, negative_lamb, positive_gamma, negative_gamma):
    k = len(negative_sample_embeddings)
    positive_loss = -F.logsigmoid(positive_gamma - torch.max(d_con4(q1_pos, q2s, postive_lamb)))
    negative_loss = 0
    for q1_neg in negative_sample_embeddings:
        negative_loss += torch.sum(F.logsigmoid(torch.max(d_con4(q1_neg, q2s, negative_lamb)) - negative_gamma))
    return positive_loss - 1/k * negative_loss

def cone_loss_negation(q1_pos, q2, negative_sample_embeddings, postive_lamb, negative_lamb, positive_gamma, negative_gamma):
    pi = torch.tensor(3.14159265358979323846)
    theta_ax2, ap2 = q2
    indicator_positive = theta_ax2 >= 0
    indicator_negative = theta_ax2 < 0

    theta_ax2[indicator_positive] = theta_ax2[indicator_positive] - pi
    theta_ax2[indicator_negative] = theta_ax2[indicator_negative] + pi

    ap2 = (2*pi) - ap2
    return cone_loss3(q1_pos, (theta_ax2, ap2), negative_sample_embeddings, postive_lamb, negative_lamb, positive_gamma, negative_gamma)


def cone_loss_skewed(q1_pos, q2, negative_sample_embeddings, lamb, gamma=12.0):
    k = len(negative_sample_embeddings)
    positive_loss = -F.logsigmoid(6-d_con3(q1_pos, q2, 0.7))
    negative_loss = 0
    for q1_neg in negative_sample_embeddings:
        negative_loss += torch.sum(F.logsigmoid(d_con3(q1_neg, q2, 0.1) - 72))
    return positive_loss - 1/k * negative_loss

def cone_loss_skewed1(q1_pos, q2, negative_sample_embeddings, lamb=0.2, gamma=12.0):
    k = len(negative_sample_embeddings)
    positive_loss = -F.logsigmoid(6-d_con3(q1_pos, q2, lamb))
    negative_loss = 0
    for q1_neg in negative_sample_embeddings:
        negative_loss += torch.sum(F.logsigmoid(d_con3(q1_neg, q2, lamb) - 36))
    return positive_loss - 1/k * negative_loss

def cone_loss_really_skewed(q1_pos, q2, negative_sample_embeddings, lamb, gamma=256):
    k = len(negative_sample_embeddings)
    positive_loss = -F.logsigmoid(1-d_con3(q1_pos, q2, lamb))
    negative_loss = 0
    for q1_neg in negative_sample_embeddings:
        negative_loss += torch.sum(F.logsigmoid(d_con3(q1_neg, q2, lamb) - gamma))
    return positive_loss - 1/k * negative_loss

if __name__ == '__main__':

    # Plotting functions
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    import matplotlib.patches as patches
    def cone_to_wedge(axis, aperture, color):
        #convert to degrees
        aperture = np.degrees(aperture)
        axis = np.degrees(axis)
        theta1 = axis - aperture/2
        theta2 = axis + aperture/2
        return patches.Wedge((0, 0), 1, theta1, theta2, fill=True, color=color, linewidth=0.5, alpha=0.5)

    def draw_cones(axis1, aperture1, axis2, aperture2):
        fig, ax = plt.subplots()

        # plot the unit circle
        circle = patches.Circle((0, 0), radius=1, fill=False, edgecolor='black', linewidth=0.5)
        ax.add_patch(circle)
        
        # Plot the first cone
        cone1 = cone_to_wedge(axis1, aperture1, color='blue')
        ax.add_patch(cone1)
        
        # Plot the second cone
        cone2 = cone_to_wedge(axis2, aperture2, color='red')
        ax.add_patch(cone2)
        
        ax.set_xlim(-max(axis1, axis2), max(axis1, axis2))
        ax.set_ylim(-max(axis1, axis2), max(axis1, axis2))
        ax.set_aspect('equal')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title('2D Unit Cones')
        ax.grid(True)
        ax.legend([cone1, cone2], ['Cone 1', 'Cone 2'])
        
        plt.show()

    V_q1 = (torch.tensor(np.pi/4), torch.tensor(np.pi/8))  # Cone 1
    V_q2 = (torch.tensor(np.pi/3), torch.tensor(np.pi/9))  # Cone 2

    # Plot the cones
    draw_cones(V_q1[0], V_q1[1], V_q2[0], V_q2[1])

    lamb = 0.0
    print("d_con: ", d_con(V_q1, V_q2, lamb, False))


    deg_to_rad = torch.tensor(torch.pi / 180)
    V_q1 = (torch.tensor(45) * deg_to_rad, torch.tensor(30) * deg_to_rad)  # Cone 1
    V_q2 = (torch.tensor(120) * deg_to_rad, torch.tensor(30) * deg_to_rad)  # Cone 2

    # Plot the cones
    draw_cones(V_q1[0], V_q1[1], V_q2[0], V_q2[1])

    lamb = 0.0
    print("d_con: ", d_con(V_q1, V_q2, lamb, False))

    V_q1 = (torch.tensor(90) * deg_to_rad, torch.tensor(30) * deg_to_rad)  # Cone 1
    V_q2 = (torch.tensor(90) * deg_to_rad, torch.tensor(60) * deg_to_rad)  # Cone 2

    # Plot the cones
    draw_cones(V_q1[0], V_q1[1], V_q2[0], V_q2[1])

    lamb = 0.0
    print("d_con: ", d_con(V_q1, V_q2, lamb, False))

    print("d_con: ", d_con(V_q2, V_q1, lamb, False))
