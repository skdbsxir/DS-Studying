import pandas as pd
import numpy as np
import warnings
import sklearn.linear_model, statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore')


df = pd.read_csv('C:\\Users\\skdbs\\Desktop\\todoData\\KOR_suicide.csv')

# df['HDI_for_year'].plot()
# plt.scatter(df['year'], df['HDI_for_year'])
# plt.show()

# year를 index로 잡아버리면 scatter로 확인을 못하네..
# df.set_index('year', inplace=True)
print(df.head)

# 4개의 점을 지나는 선을 찾아서 값을 넣기? Curve Fitting?
# plt.scatter(df['year'], df['HDI_for_year'])
# plt.show()

"""
데이터 포인트가 현재 0.711 0.778 0.856 0.898 이렇게 있음.
scatter plot으로 봤을때 대략적인 형태는 log함수 형태랑 비슷해보였지?
"""
"""
그냥 포인트사이 대충 값만 찾는게 나을것 같다.
"""
# 산점도 위에 대략적인 회귀선 그림.
sns.regplot(x = df['year'], y = df['HDI_for_year'], fit_reg = True)
plt.show()

# 줫같은 결측값을 어떤추세로 넣어야될까..............
# 그냥 엑셀로 그리자 ㅠㅠㅠㅠㅠㅠㅠ 시간 너무 잡아먹힌다


