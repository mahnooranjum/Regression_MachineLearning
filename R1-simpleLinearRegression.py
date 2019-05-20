##############################################################################
#
#    Mahnoor Anjum
#    manomaq@gmail.com
#    References:
#        SuperDataScience,
#        Official Documentation
#
#   SIMPLE LINEAR REGRESSION
##############################################################################
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
# iloc slices by index
# .values turns dataframe into numpy object
dataset = pd.read_csv('linearRegression.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

plt.scatter(X,y)
plt.show()

# Check the shape
# X.shape
# y.shape

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3)

#============================== LINEAR REGRESSION =============================
#    Fitting Simple Linear Regression to the Training set
#    Linear Model
#    y = mx + b  
#    Parameters:	
#    fit_intercept : boolean, optional, default True
#    whether to calculate the intercept for this model. 
#    If set to False, no intercept will be used in calculations 
#    (e.g. data is expected to be already centered).
#    
#    normalize : boolean, optional, default False
#    This parameter is ignored when fit_intercept is set to False. If True, 
#    the regressors X will be normalized before regression by 
#    subtracting the mean and dividing by the l2-norm. 
#    If you wish to standardize, please use sklearn.preprocessing.StandardScaler 
#    before calling fit on an estimator with normalize=False.
#    
#    copy_X : boolean, optional, default True
#    If True, X will be copied; else, it may be overwritten.
#    
#    n_jobs : int or None, optional (default=None)
#    The number of jobs to use for the computation. This will only provide 
#    speedup for n_targets > 1 and sufficient large problems. None means 1 unless
#    in a joblib.parallel_backend context. -1 means using all processors.
#    
#    Attributes:	
#    coef_ : array, shape (n_features, ) or (n_targets, n_features)
#    Estimated coefficients for the linear regression problem. If multiple targets are passed during the fit (y 2D), this is a 2D array of shape (n_targets, n_features), while if only one target is passed, this is a 1D array of length n_features.
#    
#    intercept_ : array
#    Independent term in the linear model.


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print ("intercept: ")
print (regressor.intercept_)
print ("Coefficient ")
print (regressor.coef_)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  


# Visualising the Training set results using matplotlib
plt.scatter(X_train, y_train, color = 'blue')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.title('Simple Linear Regression Training Set')
plt.xlabel('X')
plt.ylabel('y')
plt.show()


# Visualising the Test set results using matplotlib
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Simple Linear Regression Test Set')
plt.xlabel('X')
plt.ylabel('y')
plt.show()