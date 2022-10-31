#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import packages
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# In[3]:


# loat the dataset
iris = load_iris()


# In[4]:


# Create the samples and labels
X = iris['data']
y = iris['target']


# In[10]:


# Split the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[11]:


# initialize the neural network. hidden_layer is 2, 3 neurons every layer. epoch is 100000, optimization method is adam. alpha is 1e-5
clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(3,3), random_state=1, max_iter=100000) 


# In[12]:


# Training the model
clf.fit(X_train, y_train)


# In[13]:


# Evaluate the model
clf.score(X_test, y_test)


# In[14]:


X_test[9],y_test[9]


# In[16]:


# Model predict
clf.predict(X_test[9].reshape(-1,4))


# In[18]:


X_test[16],y_test[16]


# In[19]:


clf.predict(X_test[16].reshape(-1,4))


# In[ ]:




