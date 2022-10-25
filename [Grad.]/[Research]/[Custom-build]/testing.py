import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch_geometric.nn as pyg_nn
from torch_geometric import seed_everything
from torch_geometric.datasets import Planetoid

####
# from torch_geometric.nn import MessagePassing

# class CustomGNN(MessagePassing):
#     def __init__(self):
#         super().__init__()
####

# torch.manual_seed(62)
seed_everything(62)

# TODO: Conv layer 뒤에 다른 Conv layer를 붙여서 하는게 되나?
dataset = Planetoid(root=os.getcwd() + '/dataset/cora', name='cora')

hidden_dim = 32

def accuracy(pred, label):
    """calculate accuracy (classification acc)"""
    return ((pred == label).sum() / len(label)).item()

class CustomGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.conv1 = pyg_nn.GCNConv(in_channels=input_dim, out_channels=hidden_dim, normalize=True)
        self.ln = pyg_nn.LayerNorm(in_channels=hidden_dim)
        self.conv2 = pyg_nn.GraphConv(in_channels=hidden_dim, out_channels=output_dim, aggr='mean')
        # self.conv2 = pyg_nn.GCNConv(in_channels=hidden_dim, out_channels=output_dim, normalize=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.ln(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.2)

        x = self.conv2(x, edge_index)

        embedding_node = x

        output = F.log_softmax(x, dim=1)
        
        # embedding_graph = pyg_nn.global_mean_pool(x)

        # return embedding_node, embedding_graph, output
        return embedding_node, output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = dataset[0].to(device)
model = CustomGNN(dataset.num_node_features, hidden_dim, dataset.num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.02)

# print(model)

model.train()
for epoch in range(200):
    _, output = model(data)

    loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    acc = accuracy(output[data.train_mask].argmax(dim=1), data.y[data.train_mask])
    # loss = F.nll_loss(output, data.y)
    # acc = accuracy(output.argmax(dim=1), data.y)

    loss.backward()

    optimizer.step()

    if epoch % 10 == 0:
        print(f'[Epoch {epoch:>3}] train loss : {loss:.4f} || acc = {acc:.4f}')

with torch.no_grad():
    model.eval()

    embedding_node, pred = model(data)
    pred = pred.argmax(dim=1)

    acc = accuracy(pred[data.test_mask], data.y[data.test_mask])

    print(f'Test acc : {acc:.4f}')
    print(embedding_node.shape)
    # print(embedding_graph.shape)
