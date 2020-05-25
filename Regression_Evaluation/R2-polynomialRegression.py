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
#   POLYNOMIAL REGRESSION
##############################################################################
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot(X_train, y_train, regressor, text):
    # Visualising the Training set results using matplotlib
    plt.scatter(X_train, y_train, color = 'blue')
    plt.plot(np.sort(X_train, axis=0), regressor.predict(np.sort(X_train, axis=0)), color = 'red')
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


plt.scatter(X,y, color = 'green')
plt.show()

#========================== POLYNOMIAL REGRESSION =============================
#    POLYNOMIAL REGRESSION
#    
#    Generate a new feature matrix consisting of all polynomial combinations
#    of the features with degree less than or equal to the specified degree.
#    
#    Degree:
#        4 - Degree of X i.e X*X*X*X
#        
#    y = (a4 * x^4) + (a3 * x^3) + (a2 * x^2) + (a1 * x^1) + (b * x^0)
#    Parameters:	
#    degree : integer
#    The degree of the polynomial features. Default = 2.
#    
#    interaction_only : boolean, default = False
#    If true, only interaction features are produced: features 
#    that are products of at most degree distinct input features
#     (so not x[1] ** 2, x[0] * x[2] ** 3, etc.).
#    
#    include_bias : boolean
#    If True (default), then include a bias column, the feature in 
#    which all polynomial powers are zero (i.e. a column of ones - acts as 
#    an intercept term in a linear model). 
#    
#    Attributes:	
#    powers_ : array, shape (n_output_features, n_input_features)
#    powers_[i, j] is the exponent of the jth input in the ith output.
#    
#    n_input_features_ : int
#    The total number of input features.
#    
#    n_output_features_ : int
#    The total number of polynomial output features. The number of
#    output features is computed by iterating over all suitably sized 
#    combinations of input features.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/20)


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


poly = PolynomialFeatures(degree = 2)
model = make_pipeline(poly, LinearRegression())
model.fit(X_train,y_train)
y_pred = model.predict(X_test)


plot(X_train, y_train, model, "Train")
plot(X_test, y_test, model, "Test")
evaluate(y_test,y_pred)
