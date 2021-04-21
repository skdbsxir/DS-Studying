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
#print(df)
"""
sums_by_species = df.groupby('species').sum() # 생성한 데이터 프레임에서 species에 따른 종류들을 묶어서 합을 구한다. (3가지 종류를 각각 묶어서 합)
var = 'sepal width (cm)' # 분류하고자 하는 속성은 sepal width.
sums_by_species[var].plot(kind = 'pie', fontsize=20) # 구한 합에서 sepal width속성을 plot한다. 도표는 파이차트.
plt.ylabel(var, horizontalalignment = 'left') # 도표에 이름붙이는 작업. Y축에 우리가 분류한 속성의 이름을 넣어주자.
plt.title('classiffied iris with ' + var , fontsize=25) # 도표에 제목을 붙이는 작업. 
plt.show() # 띄워준다.
# plt.savefig('iris_pie_for_one_variable.png') # 사진 파일로 저장.
# plt.close()
"""
sums_by_species = df.groupby('species').sum()
sums_by_species.plot(kind = 'pie', subplots = True, layout=(2,2), legend = False) # 모든 속성들에 대한 파이차트를 그린다. 
# subplots = True로 다른 차트를 그릴수 있게 하고, layout(2,2)로 2x2 크기로 차트를 그린다. (legend = True 하면 그림 위에 색에 따른 species값이 나옴.)
plt.title('Total samples of each species')
plt.show()


















