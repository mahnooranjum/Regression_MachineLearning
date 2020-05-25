##############################################################################
#
#    Mahnoor Anjum
#    manomaq@gmail.com
#   
#    References:
#        Official Documentation
#
##############################################################################
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset 
# iloc slices by index
# .values turns dataframe into numpy object
dataset = pd.read_csv('polyRegression.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

plt.scatter(X,y, color = 'green')
plt.show()


from sklearn.linear_model import Lasso
regressor = Lasso()
regressor.fit(X, y)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.scatter(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Lasso')
plt.xlabel('X')
plt.ylabel('y')
plt.show()