# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 06:46:42 2018

@author: Administrator
"""
import numpy as np
def loadDataSet():
    dataSet=[]
    labelSet=[]
    file=open('testSet.txt')
    for input in file.readlines():
        inputSplit=input.strip().split()
        dataSet.append([1,float(inputSplit[0]),float(inputSplit[1])])
        labelSet.append(int(inputSplit[2]))
    return dataSet,labelSet

def sigmoid(x):
    return 1/(1+np.exp(-x))

def gradDes(inputSet) 