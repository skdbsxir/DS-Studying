import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 또 다른 linear regressor? statsmodels 이용
from statsmodels.formula.api import ols
import statsmodels.api as sm

# CSV loading
df = pd.read_csv('C:\\Users\\skdbs\\Desktop\\todoData\\fixed_suicide.csv')

# index 확인. 
# print(df.columns.values)

# 목표변수 : suicides_no // 설명변수 : gdp_per_capita ($)
# shape(X) : (1056,) // shape(Y) : (1056,)
X, Y = df['gdp_per_capita ($)'], df['suicides_no']
X = X.values.reshape(-1, 1)
Y = Y.values.reshape(-1, 1)

# train/test 분할. 8:2
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

"""
# Linear Regression.
# error : 2D array를 받아야 되는데 1D array를 받음. 
 ## 입력 데이터를 reshape(-1, 1) 해서 2D로 확장시켜주면 해결가능.
 ## https://stackoverflow.com/questions/51150153/valueerror-expected-2d-array-got-1d-array-instead
linear = LinearRegression()
linear.fit(X_train, Y_train)

# print('회귀 계수 : ', linear.coef_)
preds_linear = linear.predict(X_test)
plt.scatter(preds_linear, Y_test)
"""
## 여기까지 그림 자체는 과제에서 했던거랑 비슷하게 나옴.
# 당시 문제 1) 잔차가 정규분포를 따르지 않음.
# 당시 문제 2) 각 변수와 잔차 간 그래프가 상관성이 있어보임.
# 당시 찾았던 해결책 - Robust Regression, Generalized Regression.
"""
추가 해결책? https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/
LASSO and Ridge regression are advanced forms of regression analysis that can handle multicollinearity. 
If you know how to perform linear least squares regression, you’ll be able to handle these analyses with just a little additional study.

추가 해결책? https://mindscale.kr/course/basic-stat-python/13/
"""

# LinearRegression보다 ols가 더 친절하게 잘나오고 warnings도 잘띄워준다.
linear = ols('Y ~ X', data = df).fit()
print(linear.summary())
"""
Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 7.48e+04. This might indicate that there are
strong multicollinearity or other numerical problems.
"""
# jamovi로 해서 봤던 것 처럼 strong multicollinearity가 있다고 함.

# How to draw residual plot?
# https://www.statology.org/residual-plot-python/
 ## statsmodels의 plot_regress_exog() function을 이용하면 그릴 수 있다 함.

# Define figure size
fig = plt.figure(figsize=(12, 8))

# Draw regression plots
fig = sm.graphics.plot_regress_exog(linear, 'X', fig=fig) 
# jamovi에서 해봤던 것 처럼 잔차 그래프가 요상하게 나온다!

# 그럼 이걸 이제 어떻게 해결할까...
# 봤던 해결책 중에서 GLM 을 써볼까?
# https://www.statsmodels.org/stable/glm.html

# 뭔가...뭔가.....아닌거같은데....