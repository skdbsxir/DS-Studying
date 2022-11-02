"""
Train GCN model
"""
import time
import argparse
import numpy as np
# import wandb # later

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import GCN
from utils import accuracy, calc_loss
from datasets import load_data

parser = argparse.ArgumentParser()

# Arguments
# FIXME: need to add CUDA setting 
parser.add_argument('--dataset', type=str, default='cora', help='Dataset to train')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on params)')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units (also hidden feature size)')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep_prop)')

args = parser.parse_args()

# Seed setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)

# Model build
model = GCN(
    num_feature = features.shape[1],
    num_hidden = args.hidden,
    num_class = y_train.shape[1],
    dropout = args.dropout
)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Train
def train(epoch):
    t = time.time()

    model.train()

    optimizer.zero_grad()
    
    output = model(features, adj)

    # loss_train = F.nll_loss(output, y_train, train_mask)
    # loss_train = F.cross_entropy(output, y_train, train_mask)
    loss_train = calc_loss(output, y_train, train_mask)
    acc_train = accuracy(output, y_train, train_mask)

    with torch.no_grad():
        model.eval()
        output_val = model(features, adj)
        # loss_val = F.nll_loss(output_val, y_val, val_mask)
        # loss_val = F.cross_entropy(output, y_train, train_mask)
        loss_val = calc_loss(output, y_val, val_mask)
        acc_val = accuracy(output_val, y_val, val_mask)

    print(f'Epoch : {epoch+1:04d}', f'loss_train : {loss_train:.4f}', f'acc_train : {acc_train:.4f}', f'loss_val : {loss_val:.4f}', f'acc_val : {acc_val:.4f}', f'time : {time.time()-t:.4f}s')

    loss_train.backward()
    optimizer.step()

# Test
def test():
    model.eval()
    output = model(features, adj)
    # loss_test = F.nll_loss(output, y_test, test_mask)
    loss_test = calc_loss(output, y_test, test_mask)
    acc_test = accuracy(output, y_test, test_mask)
    print('Test set results : ', f'loss : {loss_test.item():.4f}', f'accuracy : {acc_test.item():.4f}')

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print('Optimization Finished!')
print(f'Total time elapsed : {time.time() - t_total:.4f}s')

# Testing
test()