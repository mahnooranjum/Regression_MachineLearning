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

def plot(X_train, y_train, regressor, text):
    # Visualising the Training set results using matplotlib
    plt.scatter(X_train, y_train, color = 'blue')
    plt.scatter(np.sort(X_train, axis=0), regressor.predict(np.sort(X_train, axis=0)), color = 'red')
    plt.title(text)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.grid()
    plt.show()

def evaluate(y_test, y_pred):
    from sklearn import metrics  
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  

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

plot(X, y, regressor, "Model")