# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 22:20:23 2018

@author: Administrator
"""
import numpy as np
#import numpy
#import 
from os import listdir
def createDataSet():
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels


"""
分类算法
"""
def classify0(inX,dataSet,labels,k):
    dataSetSize =dataSet.shape[0]
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndicies=np.argsort(distances)
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1 #0代表了当字典中不存在此分类时取0
    sortedClassCount=sorted(classCount.items(),key=lambda s:s[1],reverse=True)#对字典类型的第2个键进行排序
    return sortedClassCount[0][0]


"""
从文件中读取数据
"""
def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines)
    returnMat=np.zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))#一行中最后一个数据表示标签，将其加入classLabelVector的末尾
        index=index+1
    return returnMat,classLabelVector
"""
进行归一化操作
"""
def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=np.zeros(dataSet.shape) #shape用于返回数组array的尺寸，参数表示某一维度上的尺寸
    m=dataSet.shape[0]  #用于返回尺寸
    normDataSet=dataSet-np.tile(minVals,(m,1))
    normDataSet=normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals

"""
用于测试分类器效果
"""
def datingClassTest():
    hoRatio=0.1
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d,the real answer is:%d"%(classifierResult,datingLabels[i]))
        if(classifierResult!=datingLabels[i]):errorCount+=1.0
    print("the total error rate is :%d"%(errorCount/numTestVecs))
    
"""
利用样本集合输出结果
"""
def classifyPerson():
    result=['not at all','in small doses','in large doses']
    datingSetMat,datingLabelsVec=file2matrix('datingTestSet2.txt')
    normData,ranges,minVals=autoNorm(datingSetMat)
    flyMiles=float(input('the fly miles earned by the person is:'))
    gameTime=float(input('the time spent on video games of the person is:'))
    iceCream=float(input('the amount of the icecream of the person is:'))
    inPut=np.array([flyMiles,gameTime,iceCream])
    resultIndice=classify0((inPut-minVals)/ranges,normData,datingLabelsVec,3)
    print('You will probably like the person : ',result[resultIndice-1])
    
"""
手写识别系统
"""    

def img2Vector(filename):
    returnVector=np.zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        imgMat=fr.readline()
        for j in range(32):
            returnVector[0,32*i+j]=int(imgMat[j])#读入数据时直接放入numpy的array中
    return returnVector

def handwritingClassTest():
    fileList=listdir("trainingDigits")
    lenOfTrain=len(fileList)  
    trainLabels=[]
    trainVect=np.zeros((lenOfTrain,1024))
    for i in range(lenOfTrain):        
        fileName=fileList[i].split('.')[0]  #得到读取的文件名称
        trainLabels.append(int(fileName.split('_')[0])) #将文件名中的分类加到列表尾部
        trainVect[i,:]=img2Vector('trainingDigits\\'+fileList[i])      #利用已有程序得到训练用的数组
    testList=listdir("testDigits")
    errCount=0.0
    lenOfTest=len(testList)
    testMat=np.zeros((1,1024))
    for i in range(lenOfTest):
        fileName=testList[i].split('.')[0]
        factOfTest=fileName.split('_')[0]
        testMat=img2Vector('testDigits\\'+testList[i])    
        predict=classify0(testMat,trainVect,trainLabels,3)
        if(predict!=int(factOfTest)):errCount+=1.0
        print('The result of the classifier is %d,the real answer is %d'%(predict,int(factOfTest)))
    errRate=errCount/float(lenOfTest)
    print('\nThe total number of errors is %d (Total number of test is %d)'%(errCount,lenOfTest))
    print('\nThe error rate is %d'%(errRate))
    
    
    