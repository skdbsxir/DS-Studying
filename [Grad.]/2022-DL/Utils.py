import os
import pandas as pd
import wandb

# Save .csv file to .rating file.
def save_to_file(df:pd.DataFrame, path):
    print(f'Saving dataframe to path : {path}')
    print(f'Columns in dataframe are : {df.columns.tolist()}')

    df.to_csv(path, header=False, index=False, sep='\t')

# TODO: Define Train, Test function
# TODO: Train함수 내에서 wandb.init(), wandb.watch(), wandb.log()
## init(project='DeepLearningProject', entity='happysky12')
## watch(model, loss_fn, log='all', log_freq=10)
## train 다 지나고 나서 log({'Epoch' : epoch, 'loss' : 손실계산식})