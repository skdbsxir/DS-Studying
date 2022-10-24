"""
(Re-build work from official PyG document)
(GraphSAGE : https://mlabonne.github.io/blog/graphsage/)

TODO: GCNConv, SAGEConv

- About PyG Data class : https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data 
    > datasets에서 받은 dataset은 길이1짜리 Data class를 가지고 있음.
    > 해당 Data class는 안에 x (feature matrix X), edge_index (adj matrix A, COO format), y (true class label), {train/val/test}_mask (mask for node task) 를 가짐. (Planetoid-Cora 기준)
        >> x : (num_nodes, num_node_features) // edge_index : (2, num_edges)
    > Data class의 edge_index를 coo/csc/csr로 변환할 수 있고, 데이터 안에 self loop가 있는지, 고립된 node가 있는지도 확인이 가능.
- PyG NeighborLoader : https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.NeighborLoader
    > GraphSAGE에서 소개한 Mini-batch Neighbor Sampling을 수행.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch_geometric.nn as pyg_nn
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader # for GraphSAGE's neigh sampler
# import example_various_utils as Utils
from example_various_utils import GCNUtils

dataset = Planetoid(root=os.getcwd() + '/dataset/cora', name='cora')

# print(dataset.num_classes) # 7 class
# print(dataset.num_features) # 1433 features
# print(dataset.num_node_features) # 1433 features
# print(len(dataset)) # 1

# print(dataset[0]) # Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
# print(dataset[0].items()) # 위 Data의 내용물 Tensor들
# print(dataset[0].x.shape[0]) # num_node : 2708
# print(dataset[0].has_isolated_nodes()) # False
# print(dataset[0].has_self_loops()) # False

hidden_dim = 32

# GCN class
class GCN(nn.Module):
    """2-layer GCN"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.GConv1 = pyg_nn.GCNConv(in_channels=input_dim, out_channels=hidden_dim)
        self.GConv2 = pyg_nn.GCNConv(in_channels=hidden_dim, out_channels=output_dim)

    def forward(self, data):
        X, edge_index = data.x, data.edge_index

        X = self.GConv1(X, edge_index) # DAD * X
        embedding = X # remember 1st layer's output (embedding after 1st layer-propagation)
        X = F.relu(X) # (DAD * X) * W = H
        X = F.dropout(X, training=self.training)

        X = self.GConv2(X, edge_index) # DAD * H
        # 따라서 총 전파식은 DAD(DADHW)W 가 됨.
        
        output = F.log_softmax(X, dim=1)

        return embedding, output

# GraphSAGE class
class GraphSAGE(nn.Module):
    """2-layer"""
    def __init__(self, data, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.GSage1 = pyg_nn.SAGEConv(in_channels=input_dim, out_channels=hidden_dim, normalize=True, aggr='mean') # Aggregator (mean, max, lstm)
        self.GSage2 = pyg_nn.SAGEConv(in_channels=hidden_dim, out_channels=output_dim, normalize=True, aggr='mean')
        self.loader = NeighborLoader(
            data=data,
            num_neighbors=[5, 10], # denotes how much neighbors are sampled for each node in each iteration.
            batch_size = 16, # total batch size
            input_nodes=data.train_mask # default = None
        )

    def forward(self, data):
        X, edge_index = data.x, data.edge_index

        X = self.GSage1(X, edge_index)
        embedding = X # remember 1st layer's output (embedding after 1st layer-propagation)
        X = F.relu(X)
        X = F.dropout(X, training=self.training)

        X = self.GSage2(X, edge_index)
        
        output = F.log_softmax(X, dim=1)

        return embedding, output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = dataset[0].to(device)

model_GCN = GCN(input_dim=dataset.num_node_features, hidden_dim=hidden_dim, output_dim=dataset.num_classes).to(device)
model_SAGE = GraphSAGE(data=data, input_dim=dataset.num_node_features, hidden_dim=hidden_dim, output_dim=dataset.num_classes).to(device)
# print(model_GCN)
# print(model_SAGE)


# embeddings_GCN = Utils.GCN_train(model_GCN, data, 200)
# test_acc_GCN = Utils.GCN_test(model_GCN, data)
embeddings_GCN = GCNUtils.GCN_train(model_GCN, data, 200)
test_acc_GCN = GCNUtils.GCN_test(model_GCN, data)
print(f'GCN test acc : {test_acc_GCN:.4f}')

# embeddings_GCN = GCN_train()
# print(embeddings_GCN.shape)