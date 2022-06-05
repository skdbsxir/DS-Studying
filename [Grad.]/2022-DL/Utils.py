import pandas as pd
import wandb
from time import time
import numpy as np
from tqdm.auto import tqdm
import torch
import matplotlib.pyplot as plt

from Dataset import MovieLensDataset
from Evaluate import evaluate_model

# Save .csv file to .rating file.
def save_to_file(df:pd.DataFrame, path):
    print(f'Saving dataframe to path : {path}')
    print(f'Columns in dataframe are : {df.columns.tolist()}')

    df.to_csv(path, header=False, index=False, sep='\t')

# Train
def train_one_epoch(model, data_loader, loss_fn, optimizer, scheduler, epoch_num, device):

    print(f'Epoch : {epoch_num+1}')

    t1 = time()
    epoch_loss = []

    model.train()

    # data -> device
    for feed_dict in tqdm(data_loader, desc='In Batch...'):
        for key in feed_dict:
            if type(feed_dict[key]) != type(None):
                feed_dict[key] = feed_dict[key].to(dtype=torch.long, device=device)

        optimizer.zero_grad()

        # get prediction
        # output is sigmoid prob.
        prediction = model(feed_dict)
        
        # get actual rating for calculate loss.
        rating = feed_dict['rating']

        # convert to float and reshape
        # [batch_size, ] -> [batch_size, 1]
        rating = rating.float().view(prediction.size())

        # calculate loss
        loss = loss_fn(prediction, rating)

        # remember loss in each epoch
        epoch_loss.append(loss.item())

        loss.backward()
        optimizer.step()
    # Schedule Optimizer
    scheduler.step()
    
    # calculate total loss in batch
    epoch_loss = np.mean(epoch_loss)
    print(f'Epoch time : {(time() - t1):.1f}s')
    print(f'Train Loss : {epoch_loss}')

    return epoch_loss

# Testing
def test(model, full_dataset:MovieLensDataset, topK):
    model.eval()
    with torch.no_grad():
        (hits, ndcgs) = evaluate_model(model, full_dataset, topK)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        print(f'[Eval] HR : {hr:.4f}, NDCG : {ndcg:.4f}')

    return hr, ndcg

# Plot history
def plot_statistics(hr_list, ndcg_list, loss_list, model_alias, path):
    plt.figure()
    hr = np.array(hr_list)
    ndcg = np.array(ndcg_list)
    loss = np.array(loss_list)

    plt.plot(hr[:, 0], hr[:, 1], linestyle='-', marker='o', label='HR')
    plt.plot(ndcg[:, 0], ndcg[:, 1], linestyle='-', marker='v', label='NDCG')
    plt.plot(loss[:, 0], loss[:, 1], linestyle='-', marker='s', label='Loss')

    plt.xlabel("Epochs")
    plt.ylabel("Values")
    plt.legend()
    plt.show()
    plt.savefig(path + model_alias + '.jpg')