import pandas as pd
import sklearn.datasets as ds
import matplotlib.pyplot as plt

bs = ds.load_boston()
#'feature_names': array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='<U7')
#'target' : MEDV
#print(bs)
df = pd.DataFrame(bs.data, columns = bs.feature_names)
df['MEDV'] = bs.target


#df.plot(x = 'CRIM', y = 'MEDV', kind = 'scatter')
df.plot(x = 'CRIM', y = 'MEDV', kind = 'scatter', logx = True)
#plt.title('Criminal Rate to normal axis')
plt.title('Criminal Rate to log axis')
plt.xlabel("Criminal Occured")
plt.ylabel("House Price")
plt.show()
plt.close()









