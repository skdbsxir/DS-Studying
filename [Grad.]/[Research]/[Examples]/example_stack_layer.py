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

# From CS224W course 
# Just stack GNN layers
class GNNStack(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GNNStack, self).__init__()

        self.convs = nn.ModuleList()