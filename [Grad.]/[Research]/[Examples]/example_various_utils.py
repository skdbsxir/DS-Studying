"""
split train/test function from 'example_various.py'

- node task : We use 1 graph, so we use masking to loss & acc.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.data import Data

def accuracy(pred, label):
    """calculate accuracy (classification acc)"""
    return ((pred == label).sum() / len(label)).item()

class GCNUtils():
    def __init__(self):
        super().__init__()

    def GCN_train(model:nn.Module, data:Data, epochs:int):
        # build optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # store embeddings (to cat after-trained final embedding)
        embedding_list = []

        model.train()

        for epoch in range(epochs):
            optimizer.zero_grad()

            # model returns 2 (embedding, output)
            embeddings, output = model(data)
            embedding_list.append(embeddings)

            # Calculate train loss & acc
            train_loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
            train_acc = accuracy(output[data.train_mask].argmax(dim=1), data.y[data.train_mask])

            train_loss.backward()

            optimizer.step()
            
            # Calculate val loss & acc
            val_loss = F.nll_loss(output[data.val_mask], data.y[data.val_mask])
            val_acc = accuracy(output[data.val_mask].argmax(dim=1), data.y[data.val_mask])

            # just print
            if epoch % 10 == 0:
                print(f'[Epoch {epoch}] Train loss : {train_loss.item():.4f} || Train acc : {train_acc:.4f} || Val loss : {val_loss.item():.4f} || Val acc : {val_acc:.4f}')

        # Concat all embeddings to get final embedding
        embeddings = torch.cat(embedding_list, dim=0)

        return embeddings

    @torch.no_grad()
    def GCN_test(model:nn.Module, data:Data):
        model.eval()

        _, pred = model(data)
        pred = pred.argmax(dim=1)

        test_acc = accuracy(pred[data.test_mask], data.y[data.test_mask])

        return test_acc
    
