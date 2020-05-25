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
dataset = pd.read_csv('linearRegression.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

plt.scatter(X,y, color = 'green')
plt.show()


from sklearn.linear_model import PassiveAggressiveRegressor
regressor = PassiveAggressiveRegressor()
regressor.fit(X, y)

plot(X, y, regressor, "Model")