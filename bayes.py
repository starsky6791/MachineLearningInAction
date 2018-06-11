# -*- coding: utf-8 -*-
"""
Created on Tue May 15 20:29:31 2018

@author: Administrator
"""

from numpy import *

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

def trainBayes(dataSet,labelSet):
    lenthOfVocab=len(dataSet[0])
    numOfTrain=len(dataSet)
    p1=sum(labelSet)/numOfTrain
    