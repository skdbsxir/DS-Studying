"""
시계열 자료.

ToDo List (다시) (큰 주제로 보면)
    1) 연도별 추세 확인?
        > 시각화를 통해 확인해보자. -> X001-2. 에서 함.
    2) 연도별 자살률 확인
        2-1) 이 자살률을 나이/세대/성별 로 구분해서 시각화.
"""
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

df = pd.read_csv('C:\\Users\\user\\Desktop\\todoData\\fixed_suicide_3.csv')

# print(df.head()) # 컬럼을 분명 지운거같은데
df = df.drop(columns = ['Unnamed: 0'])
# print(df.head())

# 나라별 묶기
country_df = df.groupby('country')
# print(country_df.head())
print(country_df['year'].describe())