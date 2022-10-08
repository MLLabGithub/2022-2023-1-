#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import packages
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


# In[3]:


# read dataset of Iris
iris = load_iris()
X = iris.data
y = iris.target


# In[4]:


k_range = range(1,40)
k_error = []


# In[6]:


# iterate set k=1 to k=40 check the errors
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # parameter CV set the percentage of dataset, there is 5:1 of training VS. Testing
    scores = cross_val_score(knn, X, y, cv=6, scoring='accuracy')
    k_error.append(1-scores.mean())


# In[7]:


# plot the curve, x axis is K, y axis is error
plt.plot(k_range, k_error)
plt.xlabel('Value of K for KNN')
plt.ylabel('Error')
plt.show()


# In[13]:


import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

k_neighbors = 11
# load dataset of iris
iris = datasets.load_iris()
X = iris.data[:, :2] # use front 2 feature, easy to draw 2D image
print(X)
y = iris.target


# In[11]:


h = 0.2 # strid of the grid
# draw colorful graphic
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


# In[15]:


for weights in ['uniform', 'distance']:
    # create a KNN classifier instance, to fit by data
    clf = neighbors.KNeighborsClassifier(n_neighbors = k_neighbors, weights = weights)
    clf.fit(X, y)
    # draw the edge, set different color
    # draw [x_min, x_max] [y_min, y_max]
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # put the result into the colorful picture
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weight = '%s')" %(k_neighbors, weights))
    
plt.show()


# In[ ]:




