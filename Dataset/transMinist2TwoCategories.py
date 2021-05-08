#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/8 11:10
# @Author  : zhouyu
# @content : 
# @File    : transMinist2TwoCategories.py.py
# @Software: PyCharm

import os
import sys
'''
    由于大多数的统计学习方法面向二分类的任务
    我们使用minist数据集构造一个二分类数据集用于后续实践
    为了保证任务的合理性，我们将minist数据集中的数字奇偶性作为我们的分类任务
    数据集构成：
    data | label
    28*28手写数字图像对应的像素点 | 0,1标签，0表示该数字为偶数，1表示该数字为奇数
'''


def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    o = open(outf, "w")
    for i in range(n):
        # 读入标签 并转换为0-1标签
        label = ord(l.read(1))
        image = [0] if label%2 == 0 else [1]

        # 读入像素点
        for j in range(28*28):
            image.append(ord(f.read(1)))

        # 标签+像素点作为一条数据写入
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

if __name__ == '__main__':
    print(sys.path)

    convert("./mnist/t10k-images.idx3-ubyte", "./mnist/t10k-labels.idx1-ubyte",
            "./mnist/mnist_test.csv", 10000)
    convert("./mnist/train-images.idx3-ubyte", "./mnist/train-labels.idx1-ubyte",
            "./mnist/mnist_train.csv", 60000)