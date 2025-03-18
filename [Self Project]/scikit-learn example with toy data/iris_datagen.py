import pandas as pd
from sklearn.model_selection import train_test_split


data_file = './data/Iris.csv'
df = pd.read_csv(data_file)

# 컬럼 명 변경
df.drop(columns='Id', inplace=True)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# species 앞의 Iris 제거
df['species'] = df['species'].str.replace('Iris-', '')

# train, test 데이터 분리(실제 비어있는 test.csv 생성을 위함)
test_df = df.groupby('species').sample(n=10, random_state=42)   # species 별 10개씩 random sample
train_df = df.drop(test_df.index)                               # test_df의 index를 제외한 나머지를 train_df로 변환

# test_df의 speices 비우기
test_df['species'] = ''

train_df.to_csv('./data/train.csv', index=False)                # 120
test_df.to_csv('./data/test.csv', index=False)                  # 30