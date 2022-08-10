"""
GCN model 정의
"""

import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

class GCN(nn.Module):
    """GCN with 2 GraphConv layer (A * act(AXW) * W)"""

    # FIXME: 추후 Sequential 통해 layer 수를 늘려보기? 
    # def __init__(self, layers, num_feature, num_hidden, num_class, dropout):
    def __init__(self, num_feature, num_hidden, num_class, dropout):
        super(GCN, self).__init__()

        # 2 GraphConv layer
        self.GraphConv1 = GraphConvolution(num_feature, num_hidden)
        self.GraphConv2 = GraphConvolution(num_hidden, num_class)

        # # FIXME: nn.sequential?
        # self.GraphConvLayers = nn.ModuleList()

        # # Multiple Graph-Convolution layers
        # # 원 저자의 말대로라면 이게 1개. (그래야 output까지 해서 총 2개.)
        # for _, (num_feature, num_hidden) in enumerate(zip(layers[:-1], layers[1:])):
        #     self.GraphConvLayers.append(GraphConvolution(num_feature, num_hidden))

        self.output = GraphConvolution(layers[-1], num_class)

        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.GraphConv1(x, adj)) # 1st layer propagation
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.GraphConv2(x, adj) # 2nd layer propagation

        # for i, _ in enumerate(range(len(self.GraphConvLayers))):
        #     x = F.relu(self.GraphConvLayers[i](x, adj))
        #     x = F.dropout(x, self.dropout, training=self.training)

        output = F.log_softmax(x, dim=1)

        return output
