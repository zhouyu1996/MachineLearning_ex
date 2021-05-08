#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/8 14:55
# @Author  : zhouyu
# @content : 
# @File    : LR4classification.py
# @Software: PyCharm

import dataHandler
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

'''
    对输入数据进行分类预测
'''
def predict(w, x):
    wx = np.dot(w, x)
    P1 = np.exp(wx) / (1 + np.exp(wx))
    if P1 >= 0.5:
        return 1
    return 0

'''
    逻辑斯蒂回归训练过程
    达到训练的迭代上限，模型停机
    也可以当损失值的下降值小于我们设定的范围时进行停机
'''
def train_logisticRegression(data, label, iter = 200):
    # 向量扩增 这样可以将常数项写入整体的式子中
    for i in range(len(data)):
        data[i].append(1)
    data = np.array(data)
    # minist数据集中是28*28的像素 所以这个LR其实考察的是每个像素点对二分类的贡献
    w = np.zeros(data.shape[1])
    # 学习率步长
    yita = 0.001
    #迭代iter次进行随机梯度下降
    for i in range(iter):
        # 每次迭代遍历一次所有样本，进行随机梯度下降
        for j in range(data.shape[0]):
            wx = np.dot(w, data[j])
            yi = label[j]
            xi = data[j]
            #梯度上升
            w -=  yita * (-xi * yi + (np.exp(wx) * xi) / ( 1 + np.exp(wx)))
    #返回学到的w
    return w

'''
    逻辑回归的测试过程
'''
def test_logisticRegression(data, label, w):
    # 向量扩增 这样可以将常数项写入整体的式子中
    for i in range(len(data)):
        data[i].append(1)
    data = np.array(data)
    error = 0
    #对于测试集中每一个测试样本进行验证
    for i in range(len(data)):
        if label[i] != predict(w, data[i]):
            error += 1
    #返回准确率
    return 1 - error / len(data)

'''
    调用sklearn中的LR方法与我们实现的LR进行对比
'''
def train_sklearnLR(data, label):
    model = LogisticRegression(penalty='l2',  max_iter=1000)
    model.fit(data, label)
    return model


if __name__ == '__main__':

    # 获取训练集及标签
    print('====load data====')
    trainData, trainLabel = dataHandler.loadData('./Dataset/mnist/mnist_train.csv')
    testData, testLabel = dataHandler.loadData('./Dataset/mnist/mnist_train.csv')
    # 开始训练
    start = time.time()
    print('====train LR====')
    w = train_logisticRegression(trainData, trainLabel)
    # 测试集正确率
    print('====test LR====')
    accuracy = test_logisticRegression(testData, testLabel, w)
    # accuracy: 90.12 %
    print('test accuracy: %.2f%%' % (100 * accuracy))
    # 打印时间
    print('time span:', time.time() - start)

    # 注意sklearn已经集成了正则化方法，因此效果会略微好一点
    start_time = time.time()
    print('====train sklearn-LR====')
    model = train_sklearnLR(trainData, trainLabel)
    # 测试集正确率
    print('====test sklearn-LR====')
    predict = model.predict(testData)
    accuracy = metrics.accuracy_score(testLabel, predict)
    # accuracy: 90.35 %
    print('test accuracy: %.2f%%' % (100 * accuracy))
    # 打印时间
    print('time span:', time.time() - start)
