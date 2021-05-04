"""
Wikipedia 조회수 예측 예제.
1. 페이지의 일일 조회수 데이터를 내려 받는다.
2. 데이터엔 이상치가 있음. 상위 5% 백분위보다 높은 조회수는 5% 백분위로 대체. (5%는 임의로 정한 값.)
3. 요일이 조회수에 미치는 영향이 있을 것이라는 가설을 설정. 
 > 즉, 조회수가 일주일 주기로 하는 주기성을 갖는다는 가설을 설정. 이를 기반으로 조회수 데이터를 주기에 따라 분석하는 통계모델 사용.
4. 길이가 일주일인 Window 생성, 이를 이용해 특징값을 추출. 이 값을 이용해 일주일동안 조회수를 예측하는 회귀모델을 학습.
"""

import urllib, json
import pandas as pd
import numpy as np
import sklearn.linear_model, statsmodels.api as sm
import matplotlib.pyplot as plt

START_DATE = '20161010'
END_DATE = '20201012'
WINDOW_SIZE = 7
TOPIC = 'Cat'
# allagnets -> all-agents가 올바른 요청방식이라 함. 업데이트 된듯.
URL_TEMPLATE = ("https://wikimedia.org/api/rest_v1"
                "/metrics/pageviews/per-article"
                "/en.wikipedia/all-access/"
                "all-agents/%s/daily/%s/%s")

# 조회수를 불러오는 함수.
def get_time_series(topic, start, end) :
    url = URL_TEMPLATE % (topic, start, end)
    json_data = urllib.request.urlopen(url).read().decode('utf-8')
    data = json.loads(json_data)
    times = [rec['timestamp'] for rec in data['items']]
    values = [rec['views'] for rec in data['items']]
    times_formatted = pd.Series(times).map(
            lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:8])
    time_index = times_formatted.astype('datetime64')
    return pd.DataFrame(
            {'views' : values}, index=time_index)
    
# 선형 회귀 모형을 학습하는 함수
def line_slope(ss):
    X = np.arange(len(ss)).reshape((len(ss), 1))
    linear.fit(X, ss)
    return linear.coef_

# 선형 회귀 모형을 하나 생성. 이후 모형에 다양한 데이터를 계속 적용.
linear = sklearn.linear_model.LinearRegression()

df = get_time_series(TOPIC, START_DATE, END_DATE)

# 시계열 데이터 시각화
# 실제로 이상치가 좀 많이 있는걸 볼 수 있음.
df['views'].plot()
plt.title('Daily Views')
plt.show()

# 백분위를 기준으로 이상치 제거.
max_views = df['views'].quantile(0.95)
df.views[df.views > max_views] = max_views

# 7일을 주기로 데이터 분석
decomp = sm.tsa.seasonal_decompose(
        df['views'].values, freq = 7)
decomp.plot()
plt.suptitle('View analysis result')
plt.show()

# 날짜별로 과거 일주일의 평균, 최대값, 최소값 등 다양한 특징 추출 후 저장.
"""
#df['mean_1week'] = pd.rolling_mean(df['views'], WINDOW_SIZE) # 이동평균 (rolling mean)
#df['max_1week'] = pd.rolling_max(df['views'], WINDOW_SIZE)
#df['min_1week'] = pd.rolling_min(df['views'], WINDOW_SIZE)
#df['slope'] = pd.rolling_apply(df['views'], WINDOW_SIZE, line_slope)
#df['total_views_week'] = pd.rolling_sum(df['views'], WINDOW_SIZE)

"""
df['mean_1week'] = df['views'].rolling(WINDOW_SIZE).mean()
df['max_1week'] = df['views'].rolling(WINDOW_SIZE).max()
df['min_1week'] = df['views'].rolling(WINDOW_SIZE).min()
df['total_views_week'] = df['views'].rolling(WINDOW_SIZE).apply(line_slope)
df['total_views_week'] = df['views'].rolling(WINDOW_SIZE).sum()
df['day_of_week'] = df.index.astype(int) % 7
day_of_week_cols = pd.get_dummies(df['day_of_week']) # 가변수 만들기
df = pd.concat([df, day_of_week_cols], axis=1)

# 예측값 준비
df['total_views_next_week'] = list(df['total_views_week'][WINDOW_SIZE:]) + [np.nan for _ in range(WINDOW_SIZE)]
INDEP_VARS = ['mean_1week', 'max_1week', 'min_1week', 'slope'] + list(range(6))
DEP_VAR = 'totla_views_next_week'

n_records = df.dropna().shape[0]
# 여기서 오류난다. TypeError: cannot do slice indexing on <class 'pandas.core.indexes.datetimes.DatetimeIndex'> with these indexers [725.5] of <class 'float'>
test_data = df.dropna()[:n_records / 2]
train_data = df.dropna()[n_records / 2:]

linear.fit(train_data[INDEP_VARS], train_data[DEP_VAR])
test_preds_array = linear.predict(test_data[INDEP_VARS])
test_preds = pd.Series(test_preds_array, index=test_data.index)

print('예측값과 정답의 상관계수 : ', test_data[DEP_VAR].corr(test_preds))

