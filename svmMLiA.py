import numpy as np

#读入数据
def loadData(filename):
    dataFile=open(filename)
    dataMat=[]#用于储存训练样本的特征
    label=[]#用于储存样本的分类结果
    for line in dataFile.readlines():
        matStr=line.strip().split('\t')#将输入分割
        dataMat.append([float(matStr[0]),float(matStr[1])])
        label.append(float(matStr[2]))
    return dataMat,label


#用于对α2进行剪切
def clipAlpha(aj,H,L):
    if(aj>H):aj=H
    if(aj<L):aj=L
    return aj

#生成一个随机的内循环变量
def selectJrand(i,m):
    j=i
    while (j==i):
        j=int(np.random.uniform(0,m))
    return j


#SMO简化版算法核心
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix=np.mat(dataMatIn); labelMat=np.mat(classLabels).transpose()
    m,n=np.shape(dataMatrix)
    alpha=np.mat(np.zeros((m,1)))
    b=0;
    iterCount=0
    while (iterCount<maxIter):
        alphaPairsChanged=0
        for i in range(m):
            gI=float(np.multiply(alpha,labelMat).T*(dataMatrix*dataMatrix[i,:].T)+b)
            EI=gI-float(labelMat[i])#计算E1,用于计算α1new及α2new
            if((alpha[i]>0)and(labelMat[i]*EI>toler))or((alpha[i]<C)and(labelMat[i]*EI<-toler)):
            #选择α1的原则：α1不满足KKT条件，且优先选择间隔区域内（离超平面距离小于ξ的支持向量）
                j=selectJrand(i,m)
                gJ=float(np.multiply(alpha,labelMat).T*(dataMatrix*dataMatrix[j,:].T)+b)
                EJ=gJ-float(labelMat[j])#计算E2,用于计算α1new及α2new
                eta=float((dataMatrix[i,:]-dataMatrix[j,:])*(dataMatrix[i,:]-\
                          dataMatrix[j,:]).T)
                #计算η，我利用的是向量点积，而不是展开式
                if (eta<=0):
                    print('eta<=0')
                    continue
                alphaJOld=alpha[j].copy()
                alphaIOld=alpha[i].copy()
                #计算未剪切的α2，
                alpha[j]=alphaJOld+labelMat[j]*(EI-EJ)/eta     
                #确定需要剪切的约束范围，根据y的不同，约束范围也不同
                if(labelMat[i]!=labelMat[j]):
                    L=max(0,alphaJOld-alphaIOld)
                    H=min(C,C+alphaJOld-alphaIOld)
                else:
                    L=max(0,alphaJOld+alphaIOld-C)
                    H=min(C,alphaJOld+alphaIOld)
                #计算剪切后的α2
                alpha[j]=clipAlpha(alpha[j],H,L)
                if(alpha[j]-alphaJOld<0.00001):
                    print('j not moving enough!')
                    continue#如果α2基本没有变化，则跳出当前次循环，找下一个外层循环变量
                      
                #根据α2计算α1
                alpha[i]=alphaIOld+labelMat[i]*labelMat[j]*(alphaJOld-alpha[j])    
                #分别利用α1及α2计算b                
                bInew=float(-EI-labelMat[i]*float(dataMatrix[i,:]*dataMatrix[i,:].T)*\
                            (alpha[i]-alphaIOld)-labelMat[j]*float(dataMatrix[i\
                            ,:]*dataMatrix[j,:].T)*(alpha[j]-alphaJOld)+b)
                bJnew=float(-EJ-labelMat[i]*float(dataMatrix[i,:]*dataMatrix[j,:].T)*\
                            (alpha[i]-alphaIOld)-labelMat[j]*float(dataMatrix[j\
                            ,:]*dataMatrix[j,:].T)*(alpha[j]-alphaJOld)+b)
    
                #根据α1及α2的不同取值确定b
                if (alpha[i]>0)and(alpha[i]<C):  b=bInew
                elif(alpha[j]>0)and(alpha[j]<C): b=bJnew
                else:b=(bInew+bJnew)/2.0
                alphaPairsChanged+=1
                print('iterCount:%d,i:%d,pair changed %d'%(iterCount,i,alphaPairsChanged))
        if(alphaPairsChanged==0):iterCount+=1
        else:iterCount=0
        print('iteration number:%d'%(iterCount))   
    return alpha,b




#==============================================================================
   
    #以下为完整版SMO算法

#==============================================================================
#用于储存所有数据，包括惩罚因子、误差、输入样本、输入样本的标签及需要最优化的参数α、b
class optStruct:
    def __init__(self,dataMat,labelMat,C,toler,kerPara):
        self.dataMat=dataMat
        self.labelMat=labelMat
        self.C=C
        self.toler=toler
        self.m=dataMat.shape[0]        
        self.alphas=np.mat(np.zeros([self.m,1]))
        self.b=0
        self.ELabel=np.mat(np.zeros([self.m,2]))#第一列用于表示是否有效的标志，第二列给出E的值
        self.Gram=calcKernel(dataMat,self.m,kerPara)

 
            
            
            
#用于计算E
def calEK(oS,k):
    E=float(np.multiply(oS.alphas,oS.labelMat).T*(oS.dataMat*(oS.dataMat[k,:]).T)+oS.b)-float(oS.labelMat[k])   #不带核函数的Ek计算
#    E=float(np.multiply(oS.alphas,oS.labelMat).T*(oS.Gram[:,k])+oS.b)\
#            -float(oS.labelMat[k])  #带核函数的Ek计算
#    
    return E

#用于选择内层循环变量
def selectJ(i,oS,EI):
    maxdeltaE=0
    maxJ=-1
    oS.ELabel[i]=[1,EI]
    #只选择在第一次遍历时修正过的α，这样做的目的是排除了肯定满足要求的非支持向量点，从而提高效率
    EJMax=0
    nonZeros=np.nonzero(oS.ELabel[:,0])[0]
    if len(nonZeros)>1:
        for j in nonZeros:
            if (j!=i):
                    EJ= calEK(oS,j)  #不是直接调用ELabel里的现有值，因为此Ek会根据选择的点不同而不同，因此在代入时需要重新计算
                    if abs(EJ-EI)>maxdeltaE:
                        maxdeltaE=abs(EJ-EI)
                        maxJ=j
                        EJMax=EJ
        return maxJ,EJMax                
    else:
        maxJ=selectJrand(i,oS.m)
        EJ=calEK(oS,maxJ)
        print(oS.alphas[maxJ],EJ)
        print(maxJ)
    return maxJ,EJ

#用于更新E值
def updataE(oS,k):
    E=calEK(oS,k)
    oS.ELabel[k]=[1,E]


#计算核函数矩阵：Gram矩阵
def calcKernel(dataMat,m,kerPara):#kerPara代表了核函数的类型及参数
    Gram=np.mat(np.zeros([m,m]))
    if kerPara[0]=='lin':
        for j in range(m):
            Gram[:,j]=np.mat(dataMat)*np.mat(dataMat[j,:]).T
        
    elif kerPara[0]=='Gauss':        
        for i in range(m):
            for j in range(m):
                Gram[i,j]=np.exp(-np.linalg.norm(np.mat(dataMat)[i,:]-np.mat(\
                    dataMat)[j,:],ord=2)/2/kerPara[1]/kerPara[1])
        for j in range(m):
            delta=dataMat-dataMat[j,:]            
            Gram[:,j]=np.exp(-1*delta*delta/(2*kerPara[1]**2))                        
   #目前仅支持线性核函数及高斯核函数
    else:raise NameError('未被识别的核函数类型，请确认后重试！')

    return Gram    


#选择内部循环的节点，并进行α值的更新，返回值为1时说明进行了更新，否则返回0
def innerL(i,oS):
    #计算外层
    EI=calEK(oS,i)
    #不满足KKT条件时选择α
    if((oS.alphas[i]>0)and(oS.labelMat[i]*EI>oS.toler))or\
        ((oS.alphas[i]<oS.C)and(oS.labelMat[i]*EI<-oS.toler)):                      
            j,EJ=selectJ(i,oS,EI)
            alphaIOld=oS.alphas[i].copy()
            alphaJOld=oS.alphas[j].copy()
            #根据约束条件确定上下界
            if (oS.labelMat[i] != oS.labelMat[j]):
                L = max(0, oS.alphas[j] - oS.alphas[i])
                H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
            else:
                L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
                H = min(oS.C, oS.alphas[j] + oS.alphas[i])
            if L==H: 
#                print ("L==H"); 
                return 0
            eta=float((oS.dataMat[i,:]-oS.dataMat[j,:])*(oS.dataMat[i,:]-\
                          oS.dataMat[j,:]).T)  #不带核函数的η计算
#            eta=oS.Gram[i,i]+oS.Gram[j,j]-2*oS.Gram[i,j]  #带核函数的η计算
            
            if eta<=0:
                print("eta<=0")
                return 0
            oS.alphas[j]+=oS.labelMat[j]*(EI-EJ)/eta
            oS.alphas[j]=clipAlpha(oS.alphas[j],H,L)
            if(abs(oS.alphas[j]-alphaJOld)<0.00001):
#                print('α没有明显变化！')
                return 0
            updataE(oS,j)
            oS.alphas[i]+=oS.labelMat[i]*oS.labelMat[j]*(alphaJOld-oS.alphas[j])
            updataE(oS,i)

#            计算新的b值时
#            bInew=float(-EI-oS.labelMat[i]*float(oS.dataMat[i,:]*oS.dataMat[i,:].T)*\
#                            (oS.alphas[i]-alphaIOld)-oS.labelMat[j]*float(oS.dataMat[i\
#                            ,:]*oS.dataMat[j,:].T)*(oS.alphas[j]-alphaJOld)+oS.b)
#            bJnew=float(-EJ-oS.labelMat[i]*float(oS.dataMat[i,:]*oS.dataMat[j,:].T)*\
#                            (oS.alphas[i]-alphaIOld)-oS.labelMat[j]*float(oS.dataMat[j\
#                            ,:]*oS.dataMat[j,:].T)*(oS.alphas[j]-alphaJOld)+oS.b)
#            bInew = oS.b - EI- oS.labelMat[i]*(oS.alphas[i]-alphaIOld)*oS.Gram[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJOld)*oS.Gram[i,j]
#            bJnew= oS.b - EJ- oS.labelMat[i]*(oS.alphas[i]-alphaIOld)*oS.Gram[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJOld)*oS.Gram[j,j]
            
#启用高斯核函数
            bInew=float(-EI-oS.labelMat[i]*float(oS.Gram[i,i])*\
                            (oS.alphas[i]-alphaIOld)-oS.labelMat[j]*float(oS.Gram[j,j])*(oS.alphas[j]-alphaJOld)+oS.b)
            bJnew=float(-EJ-oS.labelMat[i]*float(oS.Gram[i,j])*\
                            (oS.alphas[i]-alphaIOld)-oS.labelMat[j]*float(oS.Gram[j,j])*(oS.alphas[j]-alphaJOld)+oS.b)
    
            #根据α的结果选择b值
            if (oS.alphas[i]>0)and(oS.alphas[i]<oS.C):  oS.b=float(bInew)
            elif(oS.alphas[j]>0)and(oS.alphas[j]<oS.C): oS.b=float(bJnew)
            else:oS.b=float((bInew+bJnew)/2.0)   
            return 1
    else: return 0

#SMO算法外部循环，用于选择外部循环的序号
def smoP(dataMatIn,classLabels,C=0.6,toler=0.01,maxIter=100,kerPara=('lin',0)):
    #将数据生成oS对象
    oS=optStruct(np.mat(dataMatIn),np.mat(classLabels).T,C,toler,kerPara)
    iterCount=0
    #遍历整个α的标签，True时遍历整个α
    entireSet=True
    alphaPairsChanged=0
    #迭代条件：为迭代次数小于最大迭代次数且每次迭代后进行过α值修正
    #由于初始时将所有α均设置为0，因此一开始是遍历整个α集合，而不是从边界上进行遍历
    while(iterCount<maxIter)and((alphaPairsChanged>0)or(entireSet)):
        #迭代前将α的修正次数置为0
        alphaPairsChanged=0
        if (entireSet):#遍历整个集合
            for i in range(oS.m):
                alphaPairsChanged+=innerL(i,oS )
#                w=calcW(oS.alphas,dataMatIn,classLabels)
#                plotData(dataMatIn,classLabels,w,oS.b)    
#                print('遍历了整个集合，迭代次数为：%d ,修改了第 %d 个α值，修改了 %\
#                      d 次'%(iterCount,i,alphaPairsChanged))
            iterCount+=1 #遍历一次集合迭代次数加1 
        else:
            nonBoundSet=np.nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            for i in nonBoundSet:
                alphaPairsChanged+=innerL(i,oS)
               
#                print('遍历了内部点，迭代次数为：%d i,修改了第 %d 个α值，修改了 %\
#                      d 次'%(iterCount,i,alphaPairsChanged))
            iterCount+=1 #遍历一次内部点迭代次数加1
        #由于α初始值为0，所以初始时必定是遍历整个集合
        if entireSet:entireSet=False  #遍历完整个集合后，开始遍历内部点。
        #当entireSet为假时遍历内部点，如果所有内部点都没有进行值更新，则重新遍历整个集合
        elif(alphaPairsChanged==0):entireSet=True
#        print('迭代次数为：%d'% iterCount)
    w=calcW(oS.alphas,dataMatIn,classLabels)
    plotData(dataMatIn,classLabels,w,oS.b)    
    return oS.b,oS.alphas,w

#用于计算w
def calcW(alphas,data,labelDat):
    alphasMat=np.mat(alphas)
    dataMat=np.mat(data)
    labelMat=np.mat(labelDat)
    w=np.multiply(alphasMat,labelMat.T).T*dataMat   
    return w

#用于输出支持向量机的结果
import matplotlib.pyplot as plt
def plotData(dataArr,labelArr,w,b):
    
    ran=np.linspace(0,8,50)
    x2=(-float(w[0,0])*ran-b)/float(w[0,1])
    plt.plot(ran,x2)    
    for i in range(len(labelArr)):
        if ((i==29)):
            plt.scatter(dataArr[i][0],dataArr[i][1],c='r',marker='o')
        else:
            if labelArr[i]==1:
                plt.scatter(dataArr[i][0],dataArr[i][1],c='b',marker='o')
            else:
                plt.scatter(dataArr[i][0],dataArr[i][1],c='b',marker='x')
      

#用于调试代码
#for i in range(10): 
#    dataMat,labelMat=loadData('testSet.txt')
#    b,alphas,w=smoP(dataMat,labelMat,0.6,0.001,100)
#w=calcW(alphas,dataMat,labelMat)
#plotData(dataMat,labelMat,w,b)