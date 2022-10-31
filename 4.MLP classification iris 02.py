#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# In[2]:


# loat the dataset
iris = load_iris()


# In[3]:


# Create the samples and labels
X = iris['data']
y = iris['target']


# In[4]:


# Split the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[5]:


# initialize the neural network. hidden_layer is 2, 4 neurons, 5 neurons respectively. epoch is 100000, optimization method is adam. alpha is 1e-5
clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(4,5), random_state=1, max_iter=100000) 


# In[6]:


# Training the model
clf.fit(X_train, y_train)


# In[7]:


# Evaluate the model
clf.score(X_test, y_test)


# In[8]:


X_test[9],y_test[9]


# In[9]:


# Model predict
clf.predict(X_test[9].reshape(-1,4))


# In[11]:


X_test[16],y_test[16]


# In[10]:


clf.predict(X_test[16].reshape(-1,4))


# In[12]:


[coef.shape for coef in clf.coefs_]


# In[15]:


clf.predict_proba(X_test[16].reshape(-1,4))

