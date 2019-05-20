##############################################################################
#
#    Mahnoor Anjum
#    manomaq@gmail.com
#    
#    References:
#        SuperDataScience,
#        Official Documentation
#
#
##############################################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression

# Regression Dataset
X, y = make_regression(n_samples=10000, n_features=1, noise=10)
X =X.reshape(10000)
datadict = {'X': X, 'y': y}
df = pd.DataFrame(data=datadict)
df.to_csv('linearRegression.csv')
plt.scatter(X,y)
plt.show()



# the equation x^3 + x^2 + x + 9
X = np.random.randn(10000)
randomize = np.random.randint(-100,100, size = 10000)
y = []
for i in range(10000):
    y.append((2*X[i]**3)-(40*X[i]**2)+(9*X[i])+24)

for i in range(10000):
    y[i] = y[i] + randomize[i]
    
datadict = {'X': X, 'y': y}
df = pd.DataFrame(data=datadict)
df.to_csv('polyRegression.csv')
plt.scatter(X,y)
plt.show()



# the equation x^3 + x^2 + x + 9
X = np.random.randn(10000)
randomize = np.random.randint(-50,50, size = 10000)
y = []
for i in range(10000):
    y.append((2*X[i]**3)-(4*X[i]**2)+(9*X[i])+24)

for i in range(10000):
    y[i] = y[i] + randomize[i]
    
datadict = {'X': X, 'y': y}
df = pd.DataFrame(data=datadict)
df.to_csv('poly2Regression.csv')
plt.scatter(X,y)
plt.show()


#MULTIPLE LINEAR REGRESSION

X = np.random.randn(10000)
Z = np.random.randn(10000)
randomize = np.random.randint(-50,50, size = 10000)
y = []
for i in range(10000):
    y.append((2*X[i])-(40*Z[i])+24)

for i in range(10000):
    y[i] = y[i] + randomize[i]
    
datadict = {'X': X, 'Z': Z, 'y': y}
df = pd.DataFrame(data=datadict)
df.to_csv('multipleRegression.csv')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Z, y, zdir='z', color="green", s=20, c=None, depthshade=True)
ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_zlabel("Y") 
