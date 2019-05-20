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

# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.scatter(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Random Forest Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()