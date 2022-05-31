import pandas as pd
from Utils import save_to_file
# from sklearn.model_selection import KFold

INPUT_PATH = './data/ratings.csv'

OUTPUT_PATH_TRAIN = './data/movielens.train.rating'
OUTPUT_PATH_TEST = './data/movielens.test.rating'
USER_FIELD = 'userId'

"""
df = pd.read_csv(INPUT_PATH)

# K-fold를 하면 loop마다 돌리면서 다르게 보는게 맞을 듯 하다.
cv = KFold(n_splits=5, shuffle=True, random_state=62)
for train_index, test_index in cv.split(df, None):
    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]

# 1. KFold 나누고, outer loop에서 index get.
# 2. torch.utils.data.SubsetRandomSampler 로 sampler 정의 (fold마다 무작위 데이터로 split)
# 3. DataLoader 에서 sampler = subsetrandomsampler 를 넘겨서 batch 단위로 load.
"""

# Load original ratings.csv file, and split it into train/test set.
# Split Method : Leave-one-out
## K-fold? -> should split in outer loop when training.
## train 시 K-Fold 나누기.
def get_train_test_df(transaction:pd.DataFrame):
    print(f'Size of entire dataset : {transaction.shape}')
    transaction.sort_values(by=['timestamp'], inplace=True)

    # DataFrame.duplicated : Mark duplicates as True except for the last occurance.
    # 중복되는 item 구분, leave-one-out Cross-validation 방식의 splitting mask 정의.
    last_transaction_mask = transaction.duplicated(subset={USER_FIELD}, keep='last')

    # Mask 이용, Leave-one-out
    # train : 100,226 // test : 610
    train_df = transaction[last_transaction_mask]
    test_df = transaction[~last_transaction_mask]

    return train_df, test_df

# Look dataset's statistics.
def report_stats(transaction:pd.DataFrame, train_df:pd.DataFrame, test_df:pd.DataFrame):
    total_size = transaction.shape[0]
    train_size = train_df.shape[0]
    test_size = test_df.shape[0]

    print(f'Total # of records : {total_size}')
    print(f"Train size : {train_size} // Train set's ratio : {train_size/total_size:.4f} %")
    print(f"Test size : {test_size} // Test set's ratio : {test_size/total_size:.4f} %")

def main():
    transactions = pd.read_csv(INPUT_PATH)
    transactions['rating'] = 1 # Replace user's rating to 1. (for binary classification : Like/Dislike)

    # train, test data making
    train_df, test_df = get_train_test_df(transactions)

    # save dataframe to file
    save_to_file(train_df, OUTPUT_PATH_TRAIN)
    save_to_file(test_df, OUTPUT_PATH_TEST)

    report_stats(transactions, train_df, test_df)

if __name__ == "__main__":
    main()