##############################################################################
#
#    Mahnoor Anjum
#    manomaq@gmail.com
#   
#    
#    References:
#        SuperDataScience,
#        Official Documentation
#
#   MULTIPLE LINEAR REGRESSION
##############################################################################
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Importing the dataset
dataset = pd.read_csv('multipleRegression.csv')
X = dataset.iloc[:, 1:3].values
y = dataset.iloc[:, 3].values


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], y, zdir='z', color="green", s=2)
ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_zlabel("Y") 

#===================== MULTIPLE LINEAR REGRESSION =============================
#    MULTIPLE LINEAR REGRESSION 
#    
#    X : numpy array or sparse matrix of shape [n_samples,n_features]
#    Training data
#    
#    y : numpy array of shape [n_samples, n_targets]
#    Target values. Will be cast to X’s dtype if necessary
#    
#     Fitting Simple Linear Regression to the Training set
#     Linear Model
#     y = mx + b  
#     
#     y = a1x1 + a2x2 + a3x3 + a4x4 ..... + bx0
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

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results by the predict method
y_pred = regressor.predict(X_test)


#Visualizing
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(X[:,0], X[:,1], y, zdir='z', color="green", s=2)
ax.scatter(X[:,0], X[:,1], regressor.predict(X), zdir='z', color="red", s=2)

ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_zlabel("Y") 
