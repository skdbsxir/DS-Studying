"""
split train/test function from 'example_various.py'

- node task : We use 1 graph, so we use masking to loss & acc.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.loader import NeighborLoader
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
                print(f'[Epoch {epoch:>3}] Train loss : {train_loss.item():.4f} || Train acc : {train_acc:.4f} || Val loss : {val_loss.item():.4f} || Val acc : {val_acc:.4f}')

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
    
class SAGEUtils():
    def __init__(self):
        super().__init__()

    def SAGE_train(model:nn.Module, data:Data, epochs:int):
        # build optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # store embeddings (to cat after-trained final embedding)
        embedding_list_inbatch = []
        embedding_list = []

        # random-walk based mini-batch neighbor sampling
        loader = NeighborLoader(
            data=data,
            # TODO: num_neighbors가 뭐냐. 정확히.
            num_neighbors=[5, 10], # denotes how much neighbors are sampled for each node in each iteration.
            batch_size = 64, # total batch size
            input_nodes=data.train_mask # default = None
        )

        model.train()

        for epoch in range(epochs):
            # GraphSAGE uses mini batch, so defince total loss & acc 
            total_loss = 0
            acc = 0
            val_loss = 0
            val_acc = 0

            # Train in mini-batch
            for batch in loader:
                optimizer.zero_grad()
                
                #FIXME: embedding_list를 여기서 구하고 GCN처럼 어떻게 전체 embedding을 구할까? 우선 다 하고나서 shape을 찍어보자.
                # GCN : [541600, 32]
                # SAGE : [31271898, 32]
                #FIXME: sage on graph-level task 한번 찾아보고, embedding을 어떻게 구해서 활용하는지 찾아볼 것.
                embeddings, output = model(batch)
                embedding_list_inbatch.append(embeddings)

                train_loss = F.nll_loss(output[batch.train_mask], batch.y[batch.train_mask])
                acc += accuracy(output[batch.train_mask].argmax(dim=1), batch.y[batch.train_mask])

                total_loss += train_loss.item()

                train_loss.backward()

                optimizer.step()

                # validation
                val_loss += F.nll_loss(output[batch.val_mask], batch.y[batch.val_mask])
                val_acc += accuracy(output[batch.val_mask].argmax(dim=1), batch.y[batch.val_mask])
            
            embedding_list.append(torch.cat(embedding_list_inbatch, dim=0)) # 각 batch에서의 cat된 embedding을 list에 저장. -> 후에 cat해서 최종 embedding get? 

            if epoch % 10 == 0:
                print(f'[Epoch {epoch:>3}] Train loss : {total_loss/len(loader):.4f} || Train acc : {acc/len(loader):.4f} || Val loss : {val_loss/len(loader):.4f} || Val acc : {val_acc/len(loader):.4f}')

        embeddings = torch.cat(embedding_list, dim=0)

        return embeddings

    @torch.no_grad()
    def SAGET_test(model:nn.Module, data:Data):
        model.eval()

        _, pred = model(data)
        pred = pred.argmax(dim=1)

        test_acc = accuracy(pred[data.test_mask], data.y[data.test_mask])

        return test_acc
