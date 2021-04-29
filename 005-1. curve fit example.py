# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

# y = 2 + 3x^3 에 임의의 잡음을 섞은 데이터 생성.
xs = np.arange(-10, 10, 0.1)
ys = 2.0 + 3.0 * (xs ** 3) + 200*np.random.normal(size=xs.size)

# 데이터를 추정하기 위한 함수의 기본형 설정.
def calc(x, a, b) :
    return a + b * (x ** 3)

# curve fitting으로 결과 검증.
# return : 함수의 파라미터(popt), 파라미터의 공분산 추정량(pcov)
cf = curve_fit(calc, xs, ys)
best_fit_params = cf[0]
print(best_fit_params) # 실제 파라미터에 근사한 값을 찾은걸 확인할 수 있다.

# curve fitting의 성능 평가. R2 Score.
# 1에 가까우면 좋게 fit 된 것. (0.96 정도로 높게 나옴을 볼 수 있다.)
# R2 Score는 실제 모형의 분산에서 모형의 분산이 차지하는 비율
# 즉, SSR / SST (== 1 - (SSE/SST))
r2 = round(r2_score(ys, calc(xs, *best_fit_params)), 3)
print('R2 Score : ', r2)

plt.title('Curve Fitting ($R^2$ Score : %.3f)' % (r2))
plt.plot(xs, ys, label='original')
plt.plot(xs, calc(xs, *best_fit_params), label='fitted')
plt.legend()
plt.show()