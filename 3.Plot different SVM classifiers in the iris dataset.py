#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


# In[2]:


def make_meshgrid(x, y, h=.02):
    """创建要绘制的点网格
    参数
    ----------
    x: 创建网格x轴所需要的数据
    y: 创建网格y轴所需要的数据
    h: 网格大小的可选大小，可选填

    返回
    -------
    xx, yy : n维数组
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


# In[3]:


def plot_contours(ax, clf, xx, yy, **params):
    """绘制分类器的决策边界。
    参数
    ----------
    ax: matplotlib子图对象
    clf: 一个分类器
    xx: 网状网格meshgrid的n维数组
    yy: 网状网格meshgrid的n维数组
    params: 传递给contourf的参数字典，可选填
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# np.c_[xx.ravel(), yy.ravel()]： np.c_：是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，类似于pandas中的merge()。


# In[4]:


# 导入数据以便后续使用
iris = datasets.load_iris()
# 采用前两个特征。我们可以通过使用二维数据集来避免使用切片。
X = iris.data[:, :2]
y = iris.target

# 我们创建一个SVM实例并拟合数据。由于要绘制支持向量，因此我们不缩放数据
C = 1.0  # SVM正则化参数
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C, max_iter=10000),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
models = (clf.fit(X, y) for clf in models)

# 为图像设置标题
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# 设置一个2x2结构的画布
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()


# matplotlib.pyplot.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None,
#                           vmin=None, vmax=None, alpha=None, linewidths=None,
#                           verts=None, edgecolors=None, hold=None, data=None,
#                           **kwargs)
# 
# x, y对应了平面点的位置，
# s控制点大小，
# c对应颜色指示值，也就是如果采用了渐变色的话，我们设置c=x就能使得点的颜色根据点的x值变化，
# cmap调整渐变色或者颜色列表的种类
# marker控制点的形状
# alpha控制点的透明度，我喜欢在数据量大的时候设置较小的alpha值，然后调整一下s值，这样产生重叠效果使得数据的聚集特征会很好地显示出来。

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()

x = np.random.randn(10000)
y = np.random.randn(10000)
plt.scatter(x, y, c='b')
plt.scatter(x+4, y, c='r')
plt.show()


# In[4]:


fig = plt.figure()

x = np.random.randn(10000)
y = np.random.randn(10000)
plt.scatter(x, y, c='b', alpha=0.05, s=10)
plt.scatter(x+4, y, c='r', alpha=0.05, s=10)
plt.show()


# In[ ]:




