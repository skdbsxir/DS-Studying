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

"""
# 상자-수염 차트. 음..
col = 'petal length (cm)'
df['ind'] = pd.Series(df.index).apply(lambda i : i % 50)
df.pivot('ind', 'species')[col].plot(kind = 'box')
plt.show()
plt.close()
"""

# 산점도를 그려보자.
"""
df.plot(kind = 'scatter' , x = "sepal length (cm)", y = "sepal width (cm)")
plt.title("Sepal length : width")
plt.show()
plt.close()
"""

# 산점도를 구분하기 쉽게 색, 투명도, 모양을 추가.
"""
colors = ["r", "g", "b"]
markers = [".", "*", "^"]
fig, ax = plt.subplots(1,1)
for i, spec in enumerate(df['species'].unique()):
    ddf = df[df['species'] == spec]
    ddf.plot(kind = "scatter", x = "sepal width (cm)", y = "sepal length (cm)", alpha = 0.5, s = 10*(i+1), ax=ax, color = colors[i], marker = markers[i],label = spec)
    
plt.legend()
plt.show()
plt.close()
"""

# 산포 행렬을 그려보자.
from pandas.tools.plotting import scatter_matrix
scatter_matrix(df)
plt.show()





