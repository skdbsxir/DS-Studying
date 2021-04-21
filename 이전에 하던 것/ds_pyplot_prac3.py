import pandas as pd
from matplotlib import pyplot as plt
import sklearn.datasets

# 원형 데이터(csv) 파일은 Anaconda3\\lib\\site-packages\\sklearn\\datasets\\data 경로에 존재.
def get_iris_df():
    ds = sklearn.datasets.load_iris() # 사이킷 런 라이브러리에서 아이리스 데이터셋을 불러온다.
    #print(ds) 출력해서 보면 원형 데이터. 행 이름, 열 이름이 없음. 보기 좋게 다듬어줘야 한다.
    
    # 데이터를 다듬기 위해 데이터 프레임 설정.  
    # 'feature_names': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    # 'target_names': array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
    df = pd.DataFrame(ds['data'], columns = ds['feature_names'])
    code_species_map = dict(zip(range(3), ds['target_names']))
    df['species'] = [code_species_map[c] for c in ds['target']] 
    return df

df = get_iris_df()
# 이번엔 히스토 그램을 그려보자. kind = 'hist'로 해주면 됌 ㅎㅎ

df.plot(kind = 'hist', subplots = True, layout = (2,2))
plt.suptitle('Iris Histogram', fontsize = 20)
plt.show()
plt.close()


for spec in df['species'].unique():
    forspec = df[df['species'] == spec]
    forspec['petal length (cm)'].plot(kind = 'hist', alpha = 0.8, label = spec) # alpha는 차트의 투명도. 0에 가까울수록 투명하고, 1에 가까울수록 짙어진다.
    
plt.legend(loc = 'upper right') # 위치를 best로 하면 알아서 최적의 위치에 범례를 띄워준다.
plt.suptitle('Petal Length classified with Iris Species')
plt.show()
plt.close()



