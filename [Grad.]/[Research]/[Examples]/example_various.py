"""
(Re-build work from official PyG document)

TODO: GCNConv, SAGEConv

- About PyG Data class : https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data 
    > datasets에서 받은 dataset은 길이1짜리 Data class를 가지고 있음.
    > 해당 Data class는 안에 x (feature matrix X), edge_index (adj matrix A, COO format), y (true class label), {train/val/test}_mask (mask for node task) 를 가짐. (Planetoid-Cora 기준)
        >> x : (num_nodes, num_node_features) // edge_index : (2, num_edges)
    > Data class의 edge_index를 coo/csc/csr로 변환할 수 있고, 데이터 안에 self loop가 있는지, 고립된 node가 있는지도 확인이 가능.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch_geometric.nn as pyg_nn
from torch_geometric.datasets import Planetoid

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

        return output
    
    def loss(self, pred, label):
        return F.nll_loss(pred, label)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_GCN = GCN(input_dim=dataset.num_node_features, hidden_dim=hidden_dim, output_dim=dataset.num_classes).to(device)
# print(model_GCN)
data = dataset[0].to(device)

optimizer = optim.Adam(model_GCN.parameters(), lr=0.01)

def train():
    model_GCN.train()

    for epoch in range(500):
        optimizer.zero_grad()

        output = model_GCN(data)

        loss = model_GCN.loss(output[data.train_mask], data.y[data.train_mask])

        loss.backward()

        optimizer.step()

        if epoch % 10 == 0:
            test_acc = test()
            print(f'[Epoch {epoch}] Train loss : {loss.item():.4f} || Acc : {test_acc:.4f}')

@torch.no_grad()
def test():
    model_GCN.eval()

    pred = model_GCN(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    
    acc = int(correct) / int(data.test_mask.sum())

    return acc

train()