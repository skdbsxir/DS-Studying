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
# 이번엔 바 차트로 그려보자. kind = 'pie'를 kind = 'bar'로 바꿔주면 됌 ㅎㅎ

sums_by_species = df.groupby('species').sum()
var = 'sepal width (cm)'
sums_by_species[var].plot(kind = 'bar', fontsize = 15, rot = 30)

plt.title('classiffied iris with ' + var , fontsize=20)
plt.show()
plt.close()


sums_by_species = df.groupby('species').sum()
sums_by_species.plot(kind = 'bar', subplots = True, fontsize = 12)

plt.suptitle('Total samples of each species') # 서브 차트를 그릴 때, 차트 전체의 제목을 설정.
plt.show()
plt.close()
