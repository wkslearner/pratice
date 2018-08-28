#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from functools import reduce
from sklearn.datasets import load_iris

class Perceptron(object):

    def __init__(self,input_num):
        """
        初始化感知器参数:权重和偏置
        :param input_num: 特征维度
        """

        #初始化权重向量
        self.weight=[0.0 for i in range(input_num)]
        #初始化偏置项
        self.bias=0.0


    #功能函数
    def __str__(self):
        """
        打印学习到的权重以及偏置项
        :return:
        """
        return 'weights\t:%s\nbias\t:%f\n' % (self.weight, self.bias)


    def activator(self,input_vector):
        """
        :param input_vector:
        :return:
        """
        return  1 if input_vector > 0 else 0


    def calculate(self,input_vector):
        """
        计算输入向量，输出神经元的结果
        计算主体：activator(wx+b)
        :return:
        """

        #print('weights:%s'%self.weight)

        #计算w*b
        first_step=input_vector*self.weight
        #计算w1*x1+w2*x2
        second_step=reduce(lambda a,b:a+b,first_step)
        #激活w1*x1+w2*x2+b
        act_result=self.activator(second_step+self.bias)

        return act_result



    def train(self,input_vectors,labels,iteration_num,learning_rate):
        """
        :param input_vectors: x向量
        :param labels: 目标值
        :param iteration_num: 迭代次数
        :param learning_rate: 学习率
        :return:
        """

        for i in range(iteration_num):
            samples=zip(input_vectors,labels)
            for input_vec,label in samples:

                #计算输出
                output=self.calculate(input_vec)
                #计算误差
                delta = label - output

                #更新weight
                self.weight=self.weight+learning_rate*delta*input_vec
                # 更新bias
                self.bias += learning_rate* delta




class LinearUnit(Perceptron):
    def __init__(self, input_num):
        '''初始化线性单元，设置输入参数的个数'''
        Perceptron.__init__(self, input_num)


def get_training_dataset():
    '''
    捏造5个人的收入数据
    '''
    # 构建训练数据
    # 输入向量列表，每一项是工作年限
    input_vecs = np.array([[5], [3], [8], [1.4], [10.1]])
    # 期望的输出列表，月薪，注意要与输入一一对应
    labels = np.array([5500, 2300, 7600, 1800, 11400])
    return input_vecs, labels


def train_linear_unit():
    '''
    使用数据训练线性单元
    '''
    # 创建感知器，输入参数的特征数为1（工作年限）
    lu = LinearUnit(1)
    # 训练，迭代10轮, 学习速率为0.01
    input_vecs, labels = get_training_dataset()
    lu.train(input_vecs, labels, 10, 0.01)
    #返回训练好的线性单元
    return lu


def plot(linear_unit):
    import matplotlib.pyplot as plt
    input_vecs, labels = get_training_dataset()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(map(lambda x: x[0], input_vecs), labels)
    weights = linear_unit.weights
    bias = linear_unit.bias
    x = range(0,12,1)
    y = map(lambda x:weights[0] * x + bias, x)
    ax.plot(x, y)
    plt.show()


if __name__ == '__main__':
    '''训练线性单元'''
    linear_unit = train_linear_unit()
    # 打印训练获得的权重
    print (linear_unit)
    # 测试
    print ('Work 3.4 years, monthly salary = %.2f' % linear_unit.calculate([3.4]))
    print ('Work 15 years, monthly salary = %.2f' % linear_unit.calculate([15]))
    print ('Work 1.5 years, monthly salary = %.2f' % linear_unit.calculate([1.5]))
    print ('Work 6.3 years, monthly salary = %.2f' % linear_unit.calculate([6.3]))
    plot(linear_unit)