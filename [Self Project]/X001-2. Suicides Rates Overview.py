import pandas as pd
import numpy as np
import warnings
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn import preprocessing

warnings.filterwarnings('ignore')


df = pd.read_csv('C:\\Users\\user\\Desktop\\todoData\\fixed_suicide_3.csv')

# df['HDI_for_year'].plot()
# plt.scatter(df['year'], df['HDI_for_year'])
# plt.show()

# year를 index로 잡아버리면 scatter로 확인을 못하네..
# df.set_index('year', inplace=True)
# print(df.head)

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
"""
# 산점도 위에 대략적인 회귀선 그림.
sns.regplot(x = df['year'], y = df['HDI_for_year'], fit_reg = True)
plt.show()
"""

# 줫같은 결측값을 어떤추세로 넣어야될까..............
# 그냥 엑셀로 적당히 넣자 ㅠㅠㅠㅠㅠㅠㅠ 시간 너무 잡아먹힌다
# 사이트가 있다. 이 값들로 넣자..
 ## https://countryeconomy.com/hdi/germany
 ## https://countryeconomy.com/hdi/japan
 ## https://countryeconomy.com/hdi/south-korea
# 다채웠다
# df.set_index('year', inplace=True)
newDf = df.drop(columns = ['Unnamed: 0'])
# print(newDf.columns)
newDf.set_index('year', inplace=True)
# print(newDf)
# print(newDf.isnull().sum()) # 됐다.
"""
ToDo List (다시) (큰 주제로 보면)
    1) 연도별 추세 확인?
        > 시각화를 통해 확인해보자.
    2) 연도별 자살률 확인
        2-1) 이 자살률을 나이/세대/성별 로 구분해서 시각화
"""
GerDF = newDf[newDf['country'] == 'Germany']
JapDF = newDf[newDf['country'] == 'Japan']
KorDF = newDf[newDf['country'] == 'Republic of Korea']

#print(GerDF)
#print(JapDF)
#print(KorDF)


# fig = sns.heatmap(data = KorDF.corr(), annot=True, fmt = '.3f', cmap='Blues')
# fig.set_xticklabels(fig.get_xticklabels(), rotation = 0)
# plt.title('<Correlation of variables>')
# plt.show()
"""
X_data = KorDF[['HDI_for_year', 'population', 'gdp_per_capita_($)']]
X_data2 = KorDF[['HDI_for_year', 'population']]
target = KorDF[['suicides_no']]
myModel = sm.OLS(target, X_data)
myModel2 = sm.OLS(target, X_data2)
fittedModel = myModel.fit()
fittedModel2 = myModel2.fit()
print(fittedModel.summary()) # 결정계수가 전에 했던거보단 괜찮게 나오긴 함.
print(fittedModel2.summary())
# print(fittedModel.params) # HDI_for_year가 -829.05즘, 나머지 두개가 0.002, 0.042
# print(fittedModel2.params) # HDI가 -34.069, population이 0.000207.
"""
"""
fittedModel.resid.plot()
plt.title('Model Residual')
plt.show() # 딱봐도 나는 시계열자료입니다 하는게 맞다.
"""
"""
cmap = sns.heatmap(X_data.corr(), annot=True, fmt='.3f', cmap='Blues')
cmap.set_xticklabels(cmap.get_xticklabels(), rotation=0)
plt.show() # 이전에서 확인했듯이 HDI, gdp_per_capita가 아주높게나옴.
"""
"""
X_data = KorDF[['HDI_for_year', 'population', 'gdp_per_capita_($)']]
vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(X_data.values, i) for i in range(X_data.shape[1])]
vif['Variables'] = X_data.columns
print(vif) # HDI (9.9) > gdp (6.36) > population (4.5)

print('\n')

X_data2 = KorDF[['HDI_for_year', 'population']]
vif2 = pd.DataFrame()
vif2['VIF Factor'] = [variance_inflation_factor(X_data2.values, i) for i in range(X_data2.shape[1])]
vif2['Variables'] = X_data2.columns
print(vif2) # HDI == population (4.553737) 어케이게 똑같이나오지
"""
# https://datascienceschool.net/03%20machine%20learning/04.03%20%EC%8A%A4%EC%BC%80%EC%9D%BC%EB%A7%81.html
# Large Condition Number 이유?
 ## 1. 변수들의 단위차이로 숫자 스케일이 크게 달라서. 스케일링으로 해결
 ## 2. 다중공선성 문제. 변수선택 or PCA 활용.
# 느낌이 HDI는 소수점, population은 숫자가 매우 커서 나오는 거 같다.

# 스케일링을 한번 해보자.
"""
target = KorDF[['suicides_no']]
feature_names = KorDF[['HDI_for_year', 'population', 'gdp_per_capita_($)']]
feature_names = ["scale({})".format(name) for name in feature_names]
myModel3 = sm.OLS(target, feature_names)
result3 = myModel3.fit()
print(result3.summary())
"""

# target이랑 population이 큰 연관
# target이랑 HDI, gdp, 100k pop 순서대로 연관.
 ## 근데 HDI, 100k pop이랑 연관이 조금 있어보임
 ## HDI랑 gdp랑 많이 강한 상관관계가 있음. 
     ### 근데 생각해보면 국민 gdp가 높으니 행복지수가 높은게 당연한거같기도하고
target = KorDF[['suicides_no']]

X_data = KorDF[['HDI_for_year', 'population', 'gdp_per_capita_($)']] # R스퀘어 0.242
scaler = preprocessing.StandardScaler().fit(X_data)
X_scaled = scaler.transform(X_data)

myModel4 = sm.OLS(target, X_scaled)
result4 = myModel4.fit()
print(result4.summary()) ## 조건수 크다는 문제는 안나왔다 무야호~~~~~~~~
# 근데 R스퀘어 값이 좀....

"""
X_data = KorDF[['population']] # R스퀘어 0.583, 
scaler = preprocessing.StandardScaler().fit(X_data)
X_scaled = scaler.transform(X_data) # R스퀘어 0.171
# myModel5 = sm.OLS(target, X_scaled)
# result5 = myModel5.fit()
# print(result5.summary())
sns.regplot(x = X_scaled, y = target, fit_reg = True)
"""
# sns.regplot(x=KorDF['gdp_per_capita_($)'], y=KorDF['HDI_for_year'], fit_reg = True)

# 그림만 봐도 아주 올바르게 큰 연관이 있어보인다.
# 다중공선성 문제.
# 이래서 target ~ HDI, gdp, population 하면 모형이 안좋게 나오지.
"""
Multicollinearity affects the coefficients and p-values, 
but it does not influence the predictions, precision of the predictions, 
and the goodness-of-fit statistics. 
If your primary goal is to make predictions, 
and you don’t need to understand the role of each independent variable, 
you don’t need to reduce severe multicollinearity.

https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/

하긴 예측이 주 목적이니 공선성 문제를 크게 신경쓰지않아도 되는건가...

위 글 아래부분에서 추가적으로 말해주는 내용
 - 표준화(standardization)를 해서 데이터를 중앙으로 몰아주면 공선성이 좀 없어질수도 있다고함.
"""

# 정규화 선형회귀를 해보자 한번.
X_data = KorDF[['HDI_for_year', 'population', 'gdp_per_capita_($)']]
# X_data = sm.add_constant(X_data) # 상수항 추가

scaler = preprocessing.StandardScaler().fit(X_data)
X_scaled = scaler.transform(X_data)

myModel6 = sm.OLS(target, X_scaled)
result6 = myModel6.fit_regularized(alpha=0.01, L1_wt=0.5, refit = True) # 똑같은데???
print(result6.summary())

# fit_regularized 했는데 그냥 fit 한거랑 결과가 완전히 똑같이 나온다. 왜지