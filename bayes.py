# -*- coding: utf-8 -*-
"""
Created on Tue May 15 20:29:31 2018

@author: Administrator
"""

import numpy as np

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec


def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet|set(document) #union of the two sets
    return list(vocabSet)

def setOfWordsToVec(vocabSet,dataSet):
    returnVec=[0]*len(vocabSet)
    for word in dataSet:
        if word in vocabSet:
            returnVec[vocabSet.index(word)]=1
        else:
            print('词汇表中不存在该词汇！')
    return returnVec

def bagOfWordsToVec(vocabSet,dataSet):
    returnVec=[0]*len(vocabSet)
    for word in dataSet:
        if word in vocabSet:
            returnVec[vocabSet.index(word)]+=1
        else:
            print('词汇表中不存在该词汇！')
    return returnVec
            


def trainBayes(dataSet,labelSet):
    numOfVocab=len(dataSet[0]) #记录样本数量
    numOfTrain=len(dataSet)  #记录特征数量
    p1=sum(labelSet)/numOfTrain
    p0Vec=np.ones(numOfVocab)#拉普拉斯平滑
    p1Vec=np.ones(numOfVocab)
    p1Num=2
    p0Num=2
    for i in range(numOfTrain):
        if labelSet[i]==1:
            p1Vec+=dataSet[i]
            p1Num+=sum(dataSet[i])
        else:
            p0Vec+=dataSet[i]
            p0Num+=sum(dataSet[i])
    p1Con=np.log(p1Vec/p1Num)#将原函数转换成log函数，以消除小数相乘引入的误差
    p0Con=np.log(p0Vec/p0Num)
    return p1Con,p0Con,p1


def classifyNB(vec2Classify,p0Vec,p1Vec,p1):
    p1=sum(vec2Classify*p1Vec)+np.log(p1)
    p0=sum(vec2Classify*p0Vec)+np.log(1-p1)
#    p1=sum(p1Vec)+np.log(p1)
#    p0=sum(p0Vec)+np.log(1-p1)
    if p1>p0:
        return 1
    else:
        return 0
    
def testingNB():
    dataSet,labelSet=loadDataSet()
    vocabList=createVocabList(dataSet)
    trainSetMatrix=[]
    for word in dataSet:
        trainSetMatrix.append(setOfWordsToVec(vocabList,word))
    p1Con,p0Con,p1=trainBayes(trainSetMatrix,labelSet)
    testWord=['love','my','dalmation']
    testVec=setOfWordsToVec(vocabList,testWord)
    print('测试文本\"love my dalmation"分类为：',classifyNB(testVec,p0Con,p1Con,p1))
    testWord=['stupid','garbage']
    testVec=setOfWordsToVec(vocabList,testWord)
    print('测试向量\"stupid garbage"分类为',classifyNB(testVec,p0Con,p1Con,p1))
