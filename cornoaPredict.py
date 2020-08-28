import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv('coronacases.csv', sep=',')
data = data[['id','cases']]
print(data.head())

#prepare data
x = np.array(data['id']).reshape(-1,1)
y = np.array(data['cases']).reshape(-1,1)
plt.plot(y,'-m')
#plt.show()

from sklearn.preprocessing import PolynomialFeatures
polyfature = PolynomialFeatures(degree=3)
x = polyfature.fit_transform(x)

print(x)

#training data
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(x,y)
accuracy = model.score(x,y)
print(f'accuracy:{round(accuracy*100,3)}%')

y0 = model.predict(x)
plt.plot(y0,'--b')
plt.show()


#prediction
days = 2
print(f'preciction - cases after{days} days:',end='')
print(round(int(model.predict(polyfature.fit_transform([[242+days]])))/1000000,2),'millions')


