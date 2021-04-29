import sklearn.datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # sklearn.cross_validation --> sklearn.model_selection 으로 바뀜.
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import normalize
from sklearn.metrics import r2_score

# Dataset Loading
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
diabetes = sklearn.datasets.load_diabetes()

"""
# 데이터 셋 확인.
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(diabetes.DESCR)
print(diabetes.feature_names)
"""

# 설명/목표 변수 지정.
# 지정 전 설명 변수 정규화.
X, Y = normalize(diabetes['data']), diabetes['target']

# 지정 후 train, test set으로 분할.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.8)

# Linear Regression
# train set으로 선형회귀모형 학습.
linear = LinearRegression()
linear.fit(X_train, Y_train)

# 회귀 계수, (설명변수-목표변수 의)상관 관계, 결정계수 계산
preds_linear = linear.predict(X_test)
# https://pandas.pydata.org/docs/reference/api/pandas.Series.corr.html
corr_linear = round(pd.Series(preds_linear).corr(pd.Series(Y_test)), 3)
rsquared_linear = r2_score(Y_test, preds_linear)

print('회귀 계수 : ', linear.coef_)
plt.subplot(1, 2, 1) # nrows = 1, ncols = 2, index = 1
plt.scatter(preds_linear, Y_test)
plt.title('Linear Regression result. Correlation = %.3f // $R^2$ Score = %.3f' % (corr_linear, rsquared_linear)) # 첨자나 각종 수식은 주변을 $로 둘러싸면 된다.
plt.xlabel('Predicted')
plt.ylabel('Actual')
# 비교를 위해 y=x 추가.
plt.plot(Y_test, Y_test, 'k--')

# Lasso(L1) Regression
# 동일하게 train set으로 Lasso 회귀모형 학습.
lasso = Lasso()
lasso.fit(X_train, Y_train)

# 회귀 계수, 상관 관계, 결정계수 계산
preds_lasso = lasso.predict(X_test)
corr_lasso = round(pd.Series(preds_lasso).corr(pd.Series(Y_test)), 3)
rsquared_lasso = round(r2_score(Y_test, preds_lasso), 3)

print('Lasso 회귀 계수 : ', lasso.coef_)
plt.subplot(1, 2, 2) # nrows = 1, ncols = 2, index = 2
plt.scatter(preds_lasso, Y_test)
plt.title('Lasso(L1) Regression result. Correlation = %.3f // $R^2$ Score = %.3f' % (corr_lasso, rsquared_lasso))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.plot(Y_test, Y_test, 'k--')

plt.show()