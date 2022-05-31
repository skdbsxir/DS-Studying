import scipy.sparse as sp
import numpy as np
from torch.utils.data import Dataset, DataLoader

np.random.seed(62)

class MovieLensDataset(Dataset):
    # PyTorch 입력을 위한 Dataset구성, train을 위해 (user, item) pair 생성.

    # dataset을 읽어, 입력을 위한 형태로 초기화
    def __init__(self, file_name, num_negative_train = 5, num_negative_test = 100):
        # Matrix 형태로 train 파일 구성
        self.trainMatrix = self.load_rating_file_as_matrix(file_name + '.train.rating')

        # 사용자 수, 아이템 수는 행렬의 형태와 동일.
        self.num_users, self.num_items = self.trainMatrix.shape

        # negative sampling을 이용해 user, item, rating (train set) 구성
        self.user_input, self.item_input, self.ratings = self.get_train_instances(
            self.trainMatrix, num_negative_train
        )

        # negative sampling을 이용해 test set 구성
        self.testRatings = self.load_rating_file_as_list(file_name + '.test.rating')
        self.testNegatives = self.create_negative_file(num_samples=num_negative_test)

        assert len(self.testRatings) == len(self.testNegatives)


    # 전체 rating 수 반환
    def __len__(self):
        return len(self.user_input)
    
    
    # 데이터 sample 1개 생성 -> Dictionary 형태
    def __getitem__(self, index):
        user_id = self.user_input[index]
        item_id = self.item_input[index]
        rating = self.ratings[index]

        itemDict = {
            'user_id' : user_id,
            'item_id' : item_id,
            'rating' : rating
        }

        return itemDict

    
    # .train.rating 파일에서 train instance를 생성.
    # 생성되는 instance는 user_input, item_input, rating.
    def get_train_instances(self, train, num_negatives):
        user_input, item_input, ratings = [], [], []
        num_users, num_items = train.shape

        for (u, i) in train.keys():
            # positive sample (Like(rated) = 1)
            user_input.append(u)
            item_input.append(i)
            ratings.append(1)

            # negative sample (Dislike(Non-rated) = 0)
            for _ in range(num_negatives):
                j = np.random.randint(1, num_items)
                while (u, j) in train:
                    j = np.random.randint(1, num_items)
                user_input.append(u)
                item_input.append(i)
                ratings.append(0)
        return user_input, item_input, ratings

    # .rating 파일을 paired list로 생성
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, 'r') as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split('\t')
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item]) # paired-list
                line = f.readline()
        return ratingList
    
    # .rating 파일을 dok matrix를 생성.
    # dok matrix는 사용자-아이템 에서 0이 아닌 위치를 기록. (인접행렬과 유사)
    # 해당 위치는 (user, item) = 1.0인 부분만을 기억. 
    ## dok matrix는 기본적으로 매우 sparse.
    def load_rating_file_as_matrix(self, filename):
        num_users, num_items = 0, 0
        with open(filename, 'r') as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split('\t')
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # dok matrix 생성
        matrix = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        
        # 생성한 dok matrix에 value 삽입
        with open(filename, 'r') as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split('\t')
                user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])

                # rating이 있다면 해당 위치 기억, matrix에 삽입
                if (rating > 0):
                    matrix[user, item] = 1.0
                line = f.readline()
        return matrix

    # https://medium.com/mlearning-ai/overview-negative-sampling-on-recommendation-systems-230a051c6cd7
    # negtive sample : user's dislike(or unseen) movies.
    # 데이터의 일부분을 negative sample로 간주해 추출.
    # The process of selecting negative examples based on a certain strategy from the user’s non-interactive product set is called Negative Sampling.
    def create_negative_file(self, num_samples=100):
        negativeList = []
        for user_item_pair in self.testRatings:
            user = user_item_pair[0]
            item = user_item_pair[1]
            negatives = []

            for t in range(num_samples):
                j = np.random.randint(1, self.num_items)
                while (user, j) in self.trainMatrix or j == item:
                    j = np.random.randint(1, self.num_items)
                negatives.append(j)
            negativeList.append(negatives)
        return negativeList

"""
# Data Loading Test
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

# (0, {'user_id': tensor([484, 66, 603, 156, 84, 607, 125, 445]), 'item_id': tensor([1407, 2571, 2640, 1265, 305, 5060, 162082, 47]), 'rating': tensor([1, 0, 1, 0, 1, 0, 0, 0])})
print(next(enumerate(train_data_loader)))
# (611, 193610)
print(train.shape)
print(train)
"""