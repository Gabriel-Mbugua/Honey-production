import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# Load in out data frame
df = pd.read_csv("honeyproduction.csv")

prod_per_year = df.groupby('year').totalprod.mean().reset_index()

X = prod_per_year.year
X = X.values.reshape(-1,1)

y = prod_per_year.totalprod
y = y.values.reshape(-1,1)

regr = linear_model.LinearRegression()
regr.fit(X,y)
# print(regr.coef_)
# print(regr.intercept_)

y_predicted = regr.predict(X)

# 2050 production prediction

X_future = np.array(range(2013,2050))
X_future = X_future.reshape(-1,1)

future_predict = regr.predict(X_future)

plt.scatter(X, y)
plt.plot(X, y_predicted)
plt.plot(X_future, future_predict)
plt.xlabel("Year")
plt.ylabel("Total Honey Production")
plt.show()




















