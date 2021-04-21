# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:36:59 2021

@author: skdbs
"""
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import sklearn.datasets
from sklearn.metrics import roc_curve, auc # ROC커브, AUC값을 통한 성능 평가
from sklearn.model_selection import train_test_split # 교차검증 이용, 훈련/테스트 셋 분할
from sklearn.linear_model import LogisticRegression # 로지스틱 회귀 - 분류
from sklearn.tree import DecisionTreeClassifier # 의사결정나무 - 분류
from sklearn.ensemble import RandomForestClassifier # 앙상블 中 랜덤포레스트 - 분류
from sklearn.naive_bayes import GaussianNB # 나이브 베이지안 분류

CLASS_MAP = {
        'Logistic Regresison' : ('-', LogisticRegression()),
        'Naive-Bayesian' : ('--', GaussianNB()),
        'Decision Tree' : ('.-', DecisionTreeClassifier(max_depth=6)),
        'Random Forest' : (':', RandomForestClassifier(max_depth=6, n_estimators=10, max_features=1)),
        }

def get_iris() : 
    ds = sklearn.datasets.load_iris()
    df = pd.DataFrame(ds['data'], columns=ds['feature_names'])
    code_species_map = dict(zip(range(3), ds['target_names']))
    df['species'] = [code_species_map[c] for c in ds['target']]
    return df

df = get_iris()

# X : 학습시킬 범주 지정 // Y : 목표범주 지정 (virginica)
# 지정 후 train, test 셋으로 분할. test_size=0.8 (80%)으로 지정 시 나머지 0.2 (20%)이 train_size로 지정 됨.
X, Y = df[df.columns[:3]], (df['species']=='versicolor')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8)

for name, (line_fmt, model) in CLASS_MAP.items() :
    model.fit(X_train, Y_train)
    preds = model.predict_proba(X_test)
    pred = pd.Series(preds[:, 1])
    
    fpr, tpr, thresholds = roc_curve(Y_test, pred)
    auc_score = auc(fpr, tpr)
    label = '%s : auc = %f' % (name, auc_score)
    plt.plot(fpr, tpr, line_fmt, linewidth=5, label=label)
    
plt.legend(loc='lower right')
plt.title('Classifier Performance Comparison')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()