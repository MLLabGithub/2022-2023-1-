#!/usr/bin/env python
# coding: utf-8

# In[115]:


# import packages
from sklearn import svm
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split


# In[116]:


# load dataset
iris = load_wine()


# In[3]:


# create samples and label
X = iris['data']
y = iris['target']


# In[117]:


# Split Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=123)


# In[141]:


# create SVC object
clf1 = svm.SVC(kernel='linear')
clf2 = svm.SVC(kernel='poly', C=1, tol=1e-5, gamma=0.7)  ## 参数对精度影响很大，   C=1, tol=1e-5, gamma=0.7)
clf3 = svm.LinearSVC(C=1.0, max_iter=1000)
clf4 = svm.SVC()


# In[142]:


# Training the model
clf1.fit(X_train, y_train)     # Training the model
clf2.fit(X_train,y_train)
clf3.fit(X_train,y_train)
clf4.fit(X_train,y_train)


# In[143]:


# Evaluate the model
clf1.score(X_test, y_test), clf2.score(X_test, y_test), clf3.score(X_test, y_test), clf4.score(X_test, y_test) 


# In[145]:


y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test)
y_pred3 = clf3.predict(X_test)
y_pred4 = clf4.predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"
     % (X_test.shape[0], (y_test != y_pred1).sum()))
print("Number of mislabeled points out of a total %d points : %d"
     % (X_test.shape[0], (y_test != y_pred2).sum()))

print("Number of mislabeled points out of a total %d points : %d"
     % (X_test.shape[0], (y_test != y_pred3).sum()))


print("Number of mislabeled points out of a total %d points : %d"
     % (X_test.shape[0], (y_test != y_pred4).sum()))


# In[146]:


clf1.support_, clf2.support_,


# In[147]:


clf1.n_support_, clf2.n_support_ #, clf3.n_supports_


# In[148]:


X, y


# In[149]:


X_test[0], y_test[0]


# In[150]:


X_test[0].reshape(-1,4)


# In[151]:


clf.predict(X_test)


# In[152]:


y_test


# In[153]:


clf.predict(X_test[29].reshape(-1, 13))


# In[128]:


# view the support vectors
clf.support_vectors_


# In[154]:


clf1.support_


# In[133]:


clf1.n_support_


# In[132]:


clf1.dual_coef_, clf1.intercept_, clf1.kernel,clf1.classes_


# In[ ]:


# In[ ]:
# C=1.0
# models = svm.SVC(kernel='linear', C=C),
#          svm.LinearSVC(C=C, max_iter=10000),
#          svm.SVC(kernel='rbf', gamma=0.7, C=C),
#          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))

