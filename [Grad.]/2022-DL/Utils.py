import pandas as pd
import wandb
from time import time
import numpy as np
from tqdm.auto import tqdm
import torch

# Save .csv file to .rating file.
def save_to_file(df:pd.DataFrame, path):
    print(f'Saving dataframe to path : {path}')
    print(f'Columns in dataframe are : {df.columns.tolist()}')

    df.to_csv(path, header=False, index=False, sep='\t')

# TODO: Test function
# TODO: Train함수 내에서 wandb.init(), wandb.watch(), wandb.log() => need to look more examples
## init(project='DeepLearningProject', entity='happysky12')
## watch(model, loss_fn, log='all', log_freq=10)
## train 다 지나고 나서 log({'Epoch' : epoch, 'loss' : 손실계산식})

# Train
def train_one_epoch(model, data_loader, loss_fn, optimizer, epoch_num, device):
    wandb.init(project='DeepLearningProject', entity='happysky12')

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
    
    # calculate total loss in batch
    epoch_loss = np.mean(epoch_loss)
    print(f'Epoch time : {(time() - t1):.1f}s')
    print(f'Train Loss : {epoch_loss}')

    return epoch_loss