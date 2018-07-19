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

def grads(inputSet,labelList):
    dataMat=np.mat(inputSet)
    labelMat=np.mat(labelList).transpose()
    m,n=np.shape(dataMat)
    alpha=0.001
    iterNum=500
    weights=np.ones((n,1))
    for i in range(iterNum):
        h=sigmoid(dataMat*weights)
        costFunction=labelMat-h
        weights=weights+alpha*dataMat.transpose()*costFunction
    return weights
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        