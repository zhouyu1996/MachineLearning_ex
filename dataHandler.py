#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/8 15:21
# @Author  : zhouyu
# @content : 
# @File    : dataHandler.py
# @Software: PyCharm


'''
  加载已经处理好的奇偶二分类Mnist数据集
'''
def loadData(fileName):
    data = []
    label = []
    fr = open(fileName, 'r')

    for line in fr.readlines():
        # 对每一行数据按切割符','进行切割，返回字段列表
        curLine = line.strip().split(',')
        label.append(int(curLine[0]))
        data.append([ int(num)/255 for num in curLine[1:]])
    #返回data和label
    return data, label

'''
    标签分布
'''
def label_statis(label):
    num = [0, 0]
    for tmp in label:
        num[tmp] += 1
    print(num)


if  __name__ == '__main__':

    data, label = loadData('./Dataset/mnist/mnist_train.csv')
    label_statis(label)

    data, label = loadData('./Dataset/mnist/mnist_test.csv')
    label_statis(label)