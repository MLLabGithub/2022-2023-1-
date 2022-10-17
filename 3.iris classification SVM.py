#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# In[2]:


# load dataset
iris = load_iris()


# In[3]:


# create samples and label
X = iris['data']
y = iris['target']


# In[22]:


# Split Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=123)


# In[23]:


# create SVC object
clf = svm.SVC()


# In[24]:


# Training the model
clf.fit(X_train, y_train)     # Training the model


# In[25]:


# Evaluate the model
clf.score(X_test, y_test)


# In[26]:


X_test[0], y_test[0]


# In[27]:


clf.predict(X_test[0].reshape(-1,4))


# In[28]:


X_test[29], y_test[29]


# In[29]:


clf.predict(X_test[29].reshape(-1,4))


# In[30]:


# view the support vectors
clf.support_vectors_


clf.n_support_, clf.support_


clf.dual_coef_, clf.intercept_, clf.kernel,clf.classes_


# In[ ]:
# C=1.0
# models = svm.SVC(kernel='linear', C=C),
#          svm.LinearSVC(C=C, max_iter=10000),
#          svm.SVC(kernel='rbf', gamma=0.7, C=C),
#          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))




