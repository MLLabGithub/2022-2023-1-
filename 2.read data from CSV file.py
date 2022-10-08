#!/usr/bin/env python
# coding: utf-8

# In[3]:


# read data from CSV file.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv


# In[6]:


data = pd.read_csv('creditcard.csv')

# 31780 rows × 31 columns
# In[7]:


data = data.drop(['Time', 'Amount'], axis=1)


# In[8]:


print(data.head())


# In[10]:


X = data.loc[:, data.columns != 'Class']
y = data.loc[:, data.columns == 'Class']
# 得到Class == 1的数量
number_one = len(data[data['Class'] == 1])
# 得到Class == 1的索引
number_one_index = np.array(data[data['Class'] == 1].index)
# 得到Class == 0的索引
number_zero_index = data[data['Class'] == 0].index



# In[18]:

## 构造正负样本数一样的训练集
# 随机选取和Class == 1一样数量的      要选择的列        要选则的数量   是否替代
random_zero_index = np.random.choice(number_zero_index, number_one, replace=True)
random_zero_index = np.array(random_zero_index)
# 拼接数组
sample = np.concatenate([random_zero_index, number_one_index])
sample_data = data.loc[sample, :]  # 按照索引获取行
print(len(sample_data[sample_data['Class'] == 1]))
print('Class == 1的概率', len(sample_data[sample_data['Class'] == 1]) / len(sample_data))
print('Class == 0的概率', len(sample_data[sample_data['Class'] == 0]) / len(sample_data))
print(len(sample_data[sample_data['Class'] == 0]))
X_sample_data = sample_data.loc[:, sample_data.columns != 'Class']
y_sample_data = sample_data.loc[:, sample_data.columns == 'Class']
# rom sklearn.cross_validation import train_test_split
# Whole dataset 原始数据集  将来用原始的test集测试
# 交叉验证 先进行切分 随机切分 test_size切分比例  random_state=
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print('训练样本特征数', len(X_train))
print('训练样本测试数', len(X_test))
print('总', len(X_train) + len(X_test))
X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(x_sample_data, y_sample_data,
                                                                                test_size=0.3, random_state=0)
print('模型样本特征数', len(X_train_sample))
print('模型样本测试数', len(X_test_sample))
print('总', len(X_train_sample) + len(X_test_sample))
print(y_test_sample)


tempX = X_train_sample.values
print('type of tempX ',type(X_train_sample.values))
print("type of x_train",type(X_train_sample))


#print(X_train_sample.head(1))
#print(X_train_sample.values[1])
#print(X_train_sample)







