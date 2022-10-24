"""
Revisited from 'example_stack_layer.py'
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.transforms as T

from example_layer import CustomGCN


# Stack 2-layer GCN
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()

        # 2-lyaer GCN
        self.GConv1 = CustomGCN(in_channels=input_dim, out_channels=hidden_dim)
        self.GConv2 = CustomGCN(in_channels=hidden_dim, out_channels=output_dim)

        # post-message-passing
        self.post_message_passing = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, output_dim)
        )

        self.dropout = 0.25
    
    def forward(self, data):
        """forward pass"""

        # X : node feature
        # edge_index : adj matrix처럼 연결 정보를 담고 있음.
        X, edge_index, batch = data.x, data.edge_index, data.batch

        # nodea feature가 없음 (featureless) : 초기 feature값을 I로 설정.
        if data.num_node_features == 0:
            X = torch.ones(data.num_nodes, 1)

        # forward pass - layer 1
        X = self.GConv1(X, edge_index)
        embedding_1 = X
        X = F.relu(X)
        X = F.dropout(X, p=self.dropout, training=self.training)

        # forward pass - layer 2
        X = self.GConv2(X, edge_index)
        embedding_2 = X
        X = F.relu(X)
        X = F.dropout(X, p=self.dropout, training=self.training)

        # post message passing
        X = self.post_message_passing(X)

        output = F.log_softmax(X, dim=1)

        return embedding_1, embedding_2, output
    
    def loss(self, pred, label):
        """calculate supervised loss"""
        # (Semi)Supervised task -> classification loss
        return F.nll_loss(pred, label)