"""
CS224W

https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html#the-messagepassing-base-class
https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
https://pytorch-geometric.readthedocs.io/en/latest/notes/load_csv.html
"""
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.transforms as T
import torch_geometric.utils as pyg_utils
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.data import DataLoader
from example_model import CustomGCN

# From CS224W course 
# Just stack GNN layers
class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNStack, self).__init__()

        # conv layer를 여러개 쌓기 위해 ModuleList 사용
        self.conv_layer_list = nn.ModuleList()

        # 정의한 함수를 통해 Append (CS224W는 GIN option을 추가해서 넣었음.)
        self.conv_layer_list.append(self.build_conv_model(input_dim, output_dim))

        # # layer normalization 추가
        # # TODO: GNN에서 LayerNorm 이유? 해당 영상 다시 한번 봐보기.
        # self.lay_norm_list = nn.ModuleList()
        # self.lay_norm_list.append(nn.LayerNorm(hidden_dim))
        # self.lay_norm_list.append(nn.LayerNorm(hidden_dim))
        # for _ in range(2):
        #     self.conv_layer_list.append()

        # post-message-passing
        # TODO: 영상 다시 한번 look. 
        self.post_message_passing = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, output_dim)
        )

        self.dropout = 0.25
        self.num_layers = 3

    def build_conv_model(self, input_dim, output_dim):
        """build model by GCN layer (made from custom layer)"""
        return CustomGCN(in_channels=input_dim, out_channels=output_dim)

    def forward(self, data):
        """forward pass"""

        # X : node feature
        # edge_index : adj matrix 역할 (연결 정보를 담고 있음)
        X, edge_index, batch = data.x, data.edge_index, data.batch

        # node feature가 없는 경우 (featureless) -> 초기 feature값을 I로 설정
        if data.num_node_features == 0:
            X = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            X = self.conv_layer_list[i](X, edge_index) # DAD * X 역할. GraphConv 수행
            embedding = X # layer의 출력물 확인을 위해 embedding 변수에 따로 저장.
            X = F.relu(X)
            X = F.dropout(X, p=self.dropout, training=self.training) # TODO: Training 인자가 뭐였지?
            
            # # Layernorm 수행
            # if not i == self.num_layers - 1:
            #     X = self.lay_norm_list[i](X)

            # # FIXME: CS224W에선 graph task인 경우, global_mean_pool을 적용함.
            # if self.task == 'graph':
            #     X = pyg_nn.global_mean_pool(X, batch)

            # message passing (via FCN)
            X = self.post_message_passing(X)

            return embedding, F.log_softmax(x, dim=1)
    
    def loss(self, pred, label):
        """Calculate supervised loss"""
        # (Semi)Supervised task이므로, classification task를 통해 loss 계산.
        return F.nll_loss(pred, label)