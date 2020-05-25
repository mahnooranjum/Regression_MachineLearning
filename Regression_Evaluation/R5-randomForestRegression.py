##############################################################################
#
#    Mahnoor Anjum
#    manomaq@gmail.com
#   
#    References:
#        SuperDataScience,
#        Official Documentation
#
#   RANDOM FOREST REGRESSION
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

plt.scatter(X,y, color = 'green')
plt.show()


##################### RANDOM FOREST REGRESSOR #################################
#
#    A random forest regressor.
#    
#    A random forest is a meta estimator that fits a number of classifying decision 
#    trees on various sub-samples of the dataset and use averaging to improve the 
#    predictive accuracy and control over-fitting.
#    
#    n_estimators : integer, optional (default=10)
#    The number of trees in the forest.
#    


# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10)
regressor.fit(X, y)


plot(X, y, regressor, "Model")
