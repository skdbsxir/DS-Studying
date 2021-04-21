# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:20:33 2021

@author: skdbs
"""

import statsmodels.api as sm
import matplotlib.pyplot as plt 

dta = sm.datasets.co2.load_pandas().data
dta.plot()
plt.title('CO2 농도')
plt.ylabel('PPM')
plt.show()