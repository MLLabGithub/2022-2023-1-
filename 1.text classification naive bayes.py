#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 贝叶斯 文本分类
from sklearn.naive_bayes import GaussianNB
import pandas as pd


# In[2]:


# dataset
documents = [['菜品','很','一般','不','建议','在这','消费'],
             ['老板','很','闹心','坑人','建议','去','别家'],
             ['让人','惊艳','东西','口味','让人','感觉','不错'],
             ['环境','不错','孜然牛柳','很','好吃'],
             ['味道','真的','一般','环境','也','比较','拥挤'],
             ['一家','性价比','很','高','餐厅','推荐'],
             ['环境','很','不错']
            ]


# In[3]:


# definition label
classVec = [1,1,0,0,1,0,0]


# In[7]:


# count words
def create_wordsAll(documents):
    words_all = set([])
    for document in documents:
        words_all = words_all | set(document)
    words_all = list(words_all)
    return words_all


# In[8]:


# create word list
words_all = create_wordsAll(documents)


# In[33]:


print(words_all)
len(words_all)


# In[26]:


# create word to Vector
def create_wordVec(document, words_all):
    dic = {}
    for word in words_all:
        if word in document:
            dic[word] = 1
        else:
            dic[word] = 0
    return dic


# In[27]:


# word matrix
trainMatrix = []


# In[28]:


for document in documents:
    trainMatrix.append(create_wordVec(document, words_all))


# In[29]:


trainMatrix


# In[30]:


# transform matrix to data frame
df = pd.DataFrame(trainMatrix)


# In[36]:


X_train = df.iloc[:-1,:]
X_train


# In[37]:


X_train.shape


# In[38]:


# create test set
X_test = df.iloc[-1:,:]
X_test


# In[39]:


# Transform label list to series
se = pd.Series(classVec)


# In[40]:


y_train = se[:-1]
y_train


# In[41]:


# Create Test label
y_test = se[-1:]
y_test


# In[42]:


# Create bayes classificator object
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train)  # fit


# In[45]:


# prediction
y_pred.predict(X_test)


# In[46]:


y_pred.score(X_test, y_test)


# In[ ]:




