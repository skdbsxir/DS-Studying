"""
CS224W

https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html#the-messagepassing-base-class
https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
https://pytorch-geometric.readthedocs.io/en/latest/notes/load_csv.html
"""

# Train & Test & Eval 
import os

import torch
import torch.optim as optim

from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid

from example_model import GCN

#### For checking
# model = GCN(64, 32, 8)
# print(model)
# for value in model.parameters():
#     print(value.shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(dataset):
    # dataset = dataset[0].to(device)
    train_loader = DataLoader(dataset, batch_size = 64, shuffle = True)
    test_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # build GCN model
    model = GCN(input_dim=max(dataset.num_node_features, 1), hidden_dim=32, output_dim=dataset.num_classes)
    # model.to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # train
    for epoch in range(200):
        total_loss = 0
        model.train()

        # batch-training
        for batch in train_loader:
            optimizer.zero_grad()
            embedding_1, embedding_2, pred = model(batch)
            label = batch.y

            # node-classification -> masking
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]

            loss = model.loss(pred, label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(train_loader.dataset)

        if epoch % 10 == 0:
            test_acc = test(test_loader, model)
            print(f'Epoch : {epoch}, Loss : {total_loss:.4f}, Test acc : {test_acc:.4f}')

    return model

def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    
    for data in loader:
        with torch.no_grad():
            embedding_1, embedding_2, pred = model(data)
            pred = pred.argmax(dim=1)
            label = data.y

            mask = data.val_mask if is_validation else data.test_mask
            pred = pred[mask]
            label = data.y[mask]
        
        correct += pred.eq(label).sum().item()
    
    total = 0
    for data in loader.dataset:
        total += torch.sum(data.test_mask).item()
    
    return correct / total

dataset = Planetoid(root=os.getcwd() + '/dataset/cora', name='cora')
model = train(dataset)