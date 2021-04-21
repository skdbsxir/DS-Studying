# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 13:58:29 2021

@author: skdbs
"""

import pandas as pd
from matplotlib import pyplot as plt
import sklearn.datasets

def get_iris() : 
    ds = sklearn.datasets.load_iris()
    df = pd.DataFrame(ds['data'], columns=ds['feature_names'])
    code_species_map = dict(zip(range(3), ds['target_names']))
    df['species'] = [code_species_map[c] for c in ds['target']]
    return df

df = get_iris()

# print(df.head())

# Histogram
"""
df.plot(kind='hist', subplots=True, layout=(2,2))
plt.suptitle('Iris Histogram', fontsize=20)
plt.show()
"""

# Scatter Plot
"""
df.plot(kind='scatter', x='sepal length (cm)', y='sepal width (cm)')
plt.show()
"""

# Scatter Plot - 종류 별 비교
 # for문 이용, df 안 species 컬럼 중 유일한 값들 (종류 3개)만큼 반복.
 # enumerate 이용, 반환을 {index, value}인 tuple 형태로.
colors = ['r', 'g', 'b']
markers = ['.', '*', '^']
fig, ax = plt.subplots(1,1)

for i, spec in enumerate(df['species'].unique()):
    df2 = df[df['species']==spec]
    df2.plot(kind='scatter', x='sepal length (cm)', y='sepal width (cm)',
             alpha=0.5, s=10*(i+1), ax=ax, color=colors[i], marker=markers[i], label=spec)
    
plt.legend()
plt.show()