# PyTorch import
import torch 
import torch.nn.functional as F
from torch import dropout, nn
from torch.utils.data import DataLoader

torch.manual_seed(62)

# Workspace import
from Dataset import MovieLensDataset
from Evaluate import evaluate_model
from Utils import plot_statistics, train_one_epoch, test

# Python imports
import argparse
from time import time
import numpy as np
import pickle
import wandb

# CUDA setting
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
if device == 'cuda': 
    # print(f'Using << {str(device).upper()} >>...')
    torch.cuda.manual_seed(62)

def parse_args():
    parser = argparse.ArgumentParser(description = 'Run MLP...')
    parser.add_argument('--path', nargs='?', default='./data/', 
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='movielens',
                        help='Dataset for use.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='# of epochs.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[16,32,16,8]',
                        help='Size of each layer. (Note : First layer is the concatenation of user-item embeddings. So layers[0]/2 is embedding size.)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Regularization for each layer.')
    parser.add_argument('--num_neg_train', type=int, default=4,
                        help='# of negative instance to pair with positive instance while training.')
    parser.add_argument('--num_neg_test', type=int, default=100,
                        help='# of negative instance to pair with positive instance while testing.')
    parser.add_argument('--lr', type=float, default=5e-3,
                        help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Add dropout layer after each dense layer, with p=dropout_prop.')
    return parser.parse_args()

class MLP(nn.Module):
    def __init__(self, n_users, n_items, layers = [16, 8], dropout=0):
        super().__init__()
        assert (layers[0] % 2 == 0), 'Layers must be an even number.'

        self.__alias__ = f"MLP_{layers}"
        self.__dropout__ = dropout

        # user & item embedding layer
        embedding_dim = int(layers[0] / 2)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # FC layers, BN layers를 생성할 ModuleList 정의
        self.fc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        # Hidden dense layers
        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(nn.Linear(in_size, out_size))
            self.bn_layers.append(nn.BatchNorm1d(out_size))
        
        # Output layer
        self.output_layer = nn.Linear(layers[-1], 1)
    
    # Dataset은 {user_id, item_id, rating} 로 구성된 dictionary.
    def forward(self, feed_dict):
        users = feed_dict['user_id']
        items = feed_dict['item_id']

        # Get user & item embedding
        user_embedding = self.user_embedding(users)
        item_embedding = self.item_embedding(items)

        # print(f'user embedding size : {user_embedding.shape}')
        # print(f'user embedding : {user_embedding}')
        # print(f'item embedding size : {item_embedding.shape}')
        # print(f'item embedding : {item_embedding}')
        

        # Concat 2 embeddings to single embedding.
        x = torch.cat([user_embedding, item_embedding], 1)
        # print(f'Concat embedding size : {x.shape}')
        # print(f'Concat embedding : {x}')

        # forward pass
        for i, _ in enumerate(range(len(self.fc_layers))):
            x = self.fc_layers[i](x)
            # print(f'현재 {i} 번째 FC  : {self.fc_layers[i]}')
            x = self.bn_layers[i](x)
            # print(f'현재 {i} 번째 BN : {self.bn_layers[i]}')
            x = F.relu(x)
            x = F.dropout(x, p=self.__dropout__, training=self.training)
        
        output = self.output_layer(x)
        rating = torch.sigmoid(output)
        return rating
    
    def predict(self, feed_dict):
        for key in feed_dict:
            if type(feed_dict[key]) != type(None):
                feed_dict[key] = torch.from_numpy(feed_dict[key]).to(dtype=torch.long, device=device)
        output_score = self.forward(feed_dict)
        return output_score.cpu().detach().numpy()
    
    def get_alias(self):
        return self.__alias__

"""
# Model Testing
path = './data/'
dataset = 'movielens'
num_negative_train = 5
num_negative_test = 100

full_dataset = MovieLensDataset(
    path + dataset,
    num_negative_train=num_negative_train,
    num_negative_test=num_negative_test
)
train, testRatings, testNegatives = full_dataset.trainMatrix, full_dataset.testRatings, full_dataset.testNegatives
num_users, num_items = train.shape

train_data_loader = DataLoader(full_dataset, batch_size=8, shuffle=True)

model = MLP(num_users, num_items, layers=[16,32,64,32,8])
print(model)
print('============'*6)

flag = 0
for feed_dict in train_data_loader:
    prediction = model.forward(feed_dict).flatten()
    rating = feed_dict['rating']
    flag += 1
    if flag == 1: break

print('============'*6)
print(f'Model output : {prediction}')
print(f'Actual value : {rating}')

print(model.state_dict().keys())
"""

def main():
    wandb.init()
    # get arguments from user input.
    args = parse_args()
    wandb.config.update(args)

    # Insert inputs to each arguments.
    path = args.path
    dataset = args.dataset
    layers = eval(args.layers)
    weight_decay = args.weight_decay
    num_negatives_train = args.num_neg_train
    num_negatives_test = args.num_neg_test
    dropout = args.dropout
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs

    print(f"MLP arguments : {args} ")

    full_dataset = MovieLensDataset(
        path + dataset, 
        num_negative_train = num_negatives_train,
        num_negative_test = num_negatives_test,
    )
    
    train, testRatings, testNegatives = full_dataset.trainMatrix, full_dataset.testRatings, full_dataset.testNegatives
    num_users, num_items = train.shape

    train_loader = DataLoader(
        full_dataset,
        batch_size = batch_size,
        shuffle = True
    )

    # Model build & move to GPU
    model = MLP(num_users, num_items, layers = layers, dropout = dropout).to(device)
    wandb.watch(model)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)

    hr_list = []
    ndcg_list = []
    loss_list = []

    # Train
    for epoch in range(epochs):
        epoch_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, epoch, device)
        
        loss_list.append(epoch_loss)

        # test & append
        hr, ndcg = test(model, full_dataset, topK=10) 
        hr_list.append(hr)
        ndcg_list.append(ndcg)

        wandb.log({
        "HR" : hr,
        "NDCG" : ndcg,
        "Loss" : epoch_loss
        })
    
    print(f'hr for epochs : {hr_list}')
    print(f'ndcg for epochs : {ndcg_list}')
    print(f'loss for epoch : {loss_list}')

    # plot_statistics(hr_list, ndcg_list, loss_list, model.get_alias(), './figs/')


if __name__ == "__main__":
    print(f'Using Device <<<< {str(device).upper()} >>>>')
    main()