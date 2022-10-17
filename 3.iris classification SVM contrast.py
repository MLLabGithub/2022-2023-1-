#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import
from sklearn import svm


# In[3]:


from sklearn import datasets
iris = datasets.load_iris()


# In[5]:


X_train = iris.data
y_train = iris.target


# In[6]:


clf_linear = svm.SVC(C=1.0, kernel='linear')


# In[7]:


clf_poly = svm.SVC(C=1.0, kernel='poly', degree=3)
clf_rbf = svm.SVC(C=1.0, kernel='rbf', gamma=0.5)
clf_rbf2 = svm.SVC(C=1.0, kernel='rbf', gamma=0.1)


# In[10]:


clfs = [clf_linear, clf_poly, clf_rbf, clf_rbf2]
titles = ['Linear Kernel:', 'Polynomial Kernel with Degree=3:', 'Gaussian Kernel with gamma=0.5:',
         'Gaussian Kernel with gamma=0.1:']


# In[12]:


for clf, i in zip(clfs, range(len(clfs))):
    clf.fit(X_train, y_train)
    print(titles[i], clf.score(X_train, y_train))

