"""
자료는 일단 시계열 자료라고 보는게 맞는것 같다.
시계열 분석이 필요.

# Time Series Regression 개념
https://seoncheolpark.github.io/book/_book/17-3-time-series-regression.html
https://datalabbit.tistory.com/85

# Time Series Regression Analysis 과정
https://sosoeasy.tistory.com/388

# Kaggle의 다른 사람들이 진행한 내용 (참고)
https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016/code?datasetId=85351&sortBy=voteCount&searchQuery=Regression
    Seaborn 패키지 따로 공부해보자. Kaggle 제출물들 둘러보니 많이들 쓰더라. 시각화도 matplotlib보다 깔끔하게 나오고.
"""
"""
ToDo List (크게)
    1) 연도별 추세 확인?
        > 시각화를 통해 확인해보자.
    2) 연도별 자살률 확인
        2-1) 이 자살률을 나이/세대/성별 로 구분해서 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings # KorDF.loc할때 warning무시하자. 원본에 영향미쳐서 뜨는 경고라는데, 원본에 영향을 미치려고 한거니까...
 
warnings.filterwarnings('ignore')

df = pd.read_csv('C:\\Users\\skdbs\\Desktop\\todoData\\fixed_suicide.csv')

# index 확인
# print(df.columns)

# 전처리를 다시 해서 깔끔하게 구분해볼까? 띄어쓰기 구분을 못했으니까.
# country / year / sex / age / suicides_no / population / suicides/100k_pop / country-year / HDI_for_year / gdp_for_year_($) / gdp_per_capita_($) / generation

# print(df['Unnamed: 0']) # 불필요한 컬럼. 이전에 수정할때 데이터 순서 index가 안지워진 듯.
newDf = df.drop(columns = ['Unnamed: 0']) # column 삭제.
# print(newDf.columns)

# index 이름 변경. rename() 메서드.
newDf = newDf.rename(columns = {'suicidex/100k pop' : 'suicides/100k_pop', 'HDI for year' : 'HDI_for_year', 'gdp_for_year ($)' : 'gdp_for_year_($)',
                                'gdp_per_capita ($)' : 'gdp_per_capita_($)'})
# print(newDf.columns)

# csv 저장
# newDf.to_csv('C:\\Users\\skdbs\\Desktop\\todoData\\fixed_suicide_2.csv', mode='w')

# 결측값이 얼마나 있나 확인해보자.
# print(newDf.isnull().sum()) # HDI_for_year에만 828개가 있다.

# 결측값을 어떻게 처리할까
 ## 1. 먼저 국가별로 나누고 - 새 df에 groupby로 묶어서 저장해볼까?
 ## 2. 결측값을 특정 값으로 채우기 (평균값 or 그냥 복사?) vs. 그냥 HDI 지우기? 위험부담이 좀 크다.

# 나라별로 묶기
country_df = newDf.groupby(['country'])
# print(country_df.head())

# 국가별로 - HDI_for_year 를 확인. how?
# print(country_df.get_group('Germany'))
# print(country_df.get_group('Japan'))
# print(country_df.get_group('Republic of Korea'))

# 나라별 DF 생성. type은 DataFrame.
GerDF = country_df.get_group('Germany')
JapDF = country_df.get_group('Japan')
KorDF = country_df.get_group('Republic of Korea')

# print(GerDF.columns)

"""
# 각 국가별 HDI_for_year의 기술통계량을 확인해보자.
print('<Germany>')
print(GerDF['HDI_for_year'].describe())
print('<Japan>')
print(JapDF['HDI_for_year'].describe())
print('<Republic of Korea>')
print(KorDF['HDI_for_year'].describe())
"""

# Korea는 아예 없네....ㅅㅂ....
# http://hdr.undp.org/en/countries/profiles/KOR
# 여기서 대충 뽑아서 넣는거도 괜찮을듯. 1985~2015.
 ## 85년도는 자료가 없음. 90년도 부터.
 ## 1990(0.732), 2000(0.823), 2010(0.889), 2015(0.907)
# print(KorDF['year'].describe()) 

# 연도에 맞춰서, 위 데이터를 HDI_for_year column에 삽입. 
# KorDF['HDI_for_yaer'] = np.where(df['year'] == 1990, 0.732)
KorDF.loc[KorDF['year']==1990, 'HDI_for_year'] = 0.732
KorDF.loc[KorDF['year']==2000, 'HDI_for_year'] = 0.823
KorDF.loc[KorDF['year']==2010, 'HDI_for_year'] = 0.889
KorDF.loc[KorDF['year']==2015, 'HDI_for_year'] = 0.907

# 일단 값은 제대로 들어감.
# print(KorDF['HDI_for_year'].describe())
# KorDF.to_csv('C:\\Users\\skdbs\\Desktop\\todoData\\KOR_suicide.csv', mode='w')

print('<Germany>')
print(GerDF['HDI_for_year'].describe())
print('<Japan>')
print(JapDF['HDI_for_year'].describe())
print('<Republic of Korea>')
print(KorDF['HDI_for_year'].describe())

# 이제 결측값을 처리해야하는데...
# 평균치로 넣는게 제일 나을 것 같다. 소수점 네번째에서 반올림.
 # Ger : 0.838, Jap : 0.861, Kor : 0.8377
# ㄴㄴ. HDI도 연도에 따라 증가하는 추세인데, 무작정 평균치로 넣으면 좀 거시기할듯.
 # (있음) (없음) (있음) 에서 양 있음의 평균치로 일일이 넣기? 아니면 작년도 + std 해서 넣기? 이건 값이 너무 커질듯.
 # 양 있음의 평균치로 일일이 넣는게 나을 것 같다.

