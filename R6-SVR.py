##############################################################################
#
#    Mahnoor Anjum
#    manomaq@gmail.com
#    References:
#        SuperDataScience,
#        Official Documentation
#
#   SUPPORT VECTOR REGRESSION
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


'''
SUPPORT VECTOR MACHINES
Epsilon-Support Vector Regression.
The free parameters in the model are C and epsilon.

kernel : string, optional (default=’rbf’)
Specifies the kernel type to be used in the algorithm. It must be one of 
‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. 
If none is given, ‘rbf’ will be used.

degree : int, optional (default=3)
Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.

'''
# Creating the SVR object and fitting it to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting a new result via the .predict method
# Predict target values of X given a model (low-level method)

y_pred = regressor.predict(X)

plt.scatter(X, y, color = 'red')
plt.scatter(X, regressor.predict(X), color = 'blue')
plt.title('Support Vector Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
