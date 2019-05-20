##############################################################################
#
#    Mahnoor Anjum
#    manomaq@gmail.com
#  
#    References:
#        SuperDataScience,
#        Official Documentation
#
#   DECISION TREE REGRESSION
##############################################################################
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset 
# iloc slices by index
# .values turns dataframe into numpy object
dataset = pd.read_csv('poly2Regression.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

plt.scatter(X,y, color = 'green')
plt.show()

#========================== DECISION TREES ====================================

#    DECISION TREE REGRESSION
#    Parameters:	
#    criterion : string, optional (default=”mse”)
#    The function to measure the quality of a split. 
#    Supported criteria are “mse” for the mean squared error, 
#    which is equal to variance reduction as feature selection 
#    criterion and minimizes the L2 loss using the mean of each 
#    terminal node, “friedman_mse”, which uses mean squared error 
#    with Friedman’s improvement score for potential splits, 
#    and “mae” for the mean absolute error, which minimizes the
#     L1 loss using the median of each terminal node.
#    
#    splitter : string, optional (default=”best”)
#    The strategy used to choose the split at each node. 
#    Supported strategies are “best” to choose the best split and 
#    “random” to choose the best random split.
#    
#    max_depth : int or None, optional (default=None)
#    The maximum depth of the tree. If None, then nodes are expanded until all 
#    leaves are pure or until all leaves contain less than min_samples_split samples.
#    
#    min_samples_split : int, float, optional (default=2)
#    The minimum number of samples required to split an internal node:
#    
#    If int, then consider min_samples_split as the minimum number.
#    If float, then min_samples_split is a fraction and 
#    ceil(min_samples_split * n_samples) are the minimum number of samples for each split.
#    Changed in version 0.18: Added float values for fractions.
#    
#    min_samples_leaf : int, float, optional (default=1)
#    The minimum number of samples required to be at a leaf node. 
#    A split point at any depth will only be considered if it leaves 
#    at least min_samples_leaf training samples in each of the left 
#    and right branches. This may have the effect of smoothing the model, 
#    especially in regression.
#    
#    If int, then consider min_samples_leaf as the minimum number.
#    If float, then min_samples_leaf is a fraction and 
#    ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.
#    Changed in version 0.18: Added float values for fractions.
#    
#    min_weight_fraction_leaf : float, optional (default=0.)
#    The minimum weighted fraction of the sum total of weights 
#    (of all the input samples) required to be at a leaf node. 
#    Samples have equal weight when sample_weight is not provided.
#    
#    max_features : int, float, string or None, optional (default=None)
#    The number of features to consider when looking for the best split:
#    
#    If int, then consider max_features features at each split.
#    If float, then max_features is a fraction and 
#    int(max_features * n_features) features are considered at each split.
#    If “auto”, then max_features=n_features.
#    If “sqrt”, then max_features=sqrt(n_features).
#    If “log2”, then max_features=log2(n_features).
#    If None, then max_features=n_features.
#    Note: the search for a split does not stop until at least one valid 
#    partition of the node samples is found, even if it requires to 
#    effectively inspect more than max_features features.
#    
#    Attributes:	
#    feature_importances_ : array of shape = [n_features]
#    Return the feature importances.
#    
#    max_features_ : int,
#    The inferred value of max_features.
#    
#    n_features_ : int
#    The number of features when fit is performed.
#    
#    n_outputs_ : int
#    The number of outputs when fit is performed.
# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X, y)


# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.scatter(X_grid, regressor.predict(X_grid), color = 'green')
plt.title('Decision Tree Regression Prediction vs Real')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()