import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 假设为 m 个数据,每个数据有 n 个特征

# 生成不同高斯分布的数据集
def createSample(mu,sigma,p,sampleNum,oneOrzero):
    # 二维正态分布
    np.random.seed(2)
    Sigma = [[sigma[0] ** 2, p * sigma[0] * sigma[1]], [p * sigma[0] * sigma[1], sigma[1] ** 2]]  # 协方差矩阵
    x,y = np.random.multivariate_normal(mu,Sigma,sampleNum).T
    if(oneOrzero == 0):
        Y = np.zeros((sampleNum,1))
    else:
        Y = np.ones((sampleNum,1))
    return (np.vstack((np.ones((1,sampleNum)),x,y)).T,Y)

# 从 https://archive.ics.uci.edu/ml/datasets/Iris 获得的数据集，该数据集一共150个数据，3个类别，4个特征
# 该数据被缓存在本地ucidata/irisa.data文件中
# 读取时选取两个类别 即选取[51,150]之间的数据
def getSampleFromUCI(begin,end,chase2del = 0):
    df = pd.read_csv('../ucidata/iris.data')  # 加载Iris数据集作为DataFrame对象
    if chase2del == 0:
        X = df.iloc[:, 0:4].values    # 取出4个特征，并把它们用Numpy数组表示
    else:
        X = df.iloc[:, [0, 2]].values # 当第三个参数设置不为0时就表示只取2个特征，可用于图像上展示

    s1 = X[50+begin:50+end]              # 取end-begin个 1数据和 0数据
    x = np.ones((np.shape(s1)[0], 1))    # 生成 X 前面的全 1 列
    y1 = np.ones((np.shape(s1)[0],1))    # 生成 Y 列
    x1 = np.hstack((x, s1))

    s2 = X[100+begin:100+end]
    x = np.ones((np.shape(s2)[0],1))
    y2 = np.zeros((np.shape(s2)[0],1))
    x2 = np.hstack((x,s2))

    return (np.vstack((x1,x2)),np.vstack((y1,y2)))     # 返回 X 和 Y

# 对于2个特征的数据，用此函数直观表示
# 参数分别为：第一个数据集，第二个数据集，求得的 w,
# 后两个参数用于画直线图时，选取合适的两个点用于画直线
def showSample(s1,s2,x,point1=1,point2=3):
    #print(x)
    x1 = np.array([point1,point2])
    y = -(x[1,0]/x[2,0])*x1-(x[0,0]/x[2,0])
    plt.plot(x1,y)
    plt.plot(s1[:,1],s1[:,2],"x",color="blue")
    plt.plot(s2[:,1],s2[:,2],"+",color="red")
    plt.show()

# 使用梯度下降法计算W
# 传入的参数分别为:
# X, Y ,最大迭代次数 , 迭代精度 , 正则项Lambda
def caculateW_GD(X,Y,epsilon,alpha,Lambda=0,MaxLoop=1e5):
    X = getNormal(X)#对数据进行归一化处理
    w = np.zeros((np.shape(X)[1],1))                                # 生成 W(n*1)的向量
    Lw_G = lambda W :X.T*(Y-np.exp(X*W)/(1+np.exp(X*W)))-Lambda*W   # 梯度计算
    num = 0
    while num < MaxLoop:
        temp = Lw_G(w)*alpha
        if(np.fabs(temp).max()<epsilon):
            break;
        else:
            w = w + temp
        num += 1
    return (w,num)                    # 返回值用来为 ( w , 迭代次数)

# 该函数用啦计算错误率,传入的参数为：
# 测试样本 X ,对应的类别(0或1)列向量 ,参数 w 列向量
def getError(X,Y,w):
    y = X*w;                         # 得到预测值

    for i in range(np.shape(y)[0]):  # 将预测值改为0或1
        if(y[i][0] <= 0 ):
            y[i][0] = 0
        else:
            y[i][0] = 1

    y1 = Y - y                       # 实际值与预测值相减
    #print(y)
    m = 0
    for i in y1:                     # 计算有多少个预测不对的数据
        if(i != 0):
            m+=1

    return m/np.shape(y1)[0]         # 返回错误率

# 使用牛顿法计算，传入的参数为：
# 数据集 X ,实际类别 Y 向量，精度，最大迭代次数
def caculateW_NT(X,Y,epsilon = 10e-11,MaxLoop = 50,Lambda=0):
    X = getNormal(X)#对数据进行归一化处理
    sigmoid  = lambda W: np.exp(X*W)/(1.0 + np.exp(X*W))
    grad = lambda W: X.T*(Y-sigmoid(w))
    m = np.shape(X)
    w = np.zeros((m[1],1))    # 初始化 w
    i = 0
    while i <= MaxLoop:
        i+= 1
        s = sigmoid(w)
        A = np.diag(np.multiply(s, s - 1).T.A[0]) # 更新 A
        H = X.T*A*X            # 更新H
        if(np.linalg.norm(H)<1e-4 or np.isnan(np.linalg.norm(H))):
            break
        deta_w = np.linalg.pinv(H) * grad(w)    # 更新 H^-1*grad(w)
        w = w-deta_w-Lambda*w
        if np.linalg.norm(deta_w) < epsilon:
            break
    return w

# 归一化一个矩阵
def getNormal(X):
    XMin = X.min(0)
    XMin[0, 0] = 0  # 第一行全为1不改变
    XMax = X.max(0)
    X_normal =  (X - XMin) / (XMax - XMin)
    return  X_normal

#使用自己的数据进行测试
# 最后一个参数表明选择使用哪种优化函数 GD为梯度上升法，NT为牛顿法
def useMyOwnData(sampleNum = 50,trainingRate = 0.8,p = 0,printMsg=0,select="GD"):
    if select=="GD":
        print("\n使用自己的数据,且为梯度上升法")
    else:
        print("\n使用自己的数据,且为牛顿法")
    # 算法中定义的常量
    epsilon = 1e-4
    alpha = 1e-2
    Lambda = 1e-3

    # 生成数据定义的数据均值和方差
    mu = [0.2,0.2]                # 平均值
    sigma = [0.1,0.1]            # 方差
    mu2 = [0.4,0.4]
    sigma2 = [0.1,0.1]

    print("mu1",mu,"sigma1",sigma)
    print("mu2",mu2,"sigma2",sigma2)
    print("相关系数",p)
    trainingNum = int(sampleNum*trainingRate)

    # 得到第一个类别的训练和测试数据集
    s1_trainning = createSample(mu,sigma,p,trainingNum,1)
    s1_test = createSample(mu,sigma,p,sampleNum-trainingNum,1)

    # 得到第二个类别的训练的测试数据集
    s2_trainning = createSample(mu2,sigma2,p,trainingNum,0)
    s2_test = createSample(mu2,sigma2,p,sampleNum-trainingNum,0)

    # 得到 X 和 Y
    X = np.vstack((s1_trainning[0],s2_trainning[0]))
    Y = np.vstack((s1_trainning[1],s2_trainning[1]))
    X_Normal = getNormal(np.matrix(X))
    if select == "GD":
        getW = caculateW_GD(np.matrix(X_Normal),np.matrix(Y),epsilon,alpha,Lambda=Lambda)[0]
    else:
        getW = caculateW_NT(np.matrix(X_Normal),np.matrix(Y),Lambda=Lambda)
    print("w:\n",getW)
    X_test = np.vstack((s1_test[0],s2_test[0]))
    Y_test = np.vstack((s1_test[1],s2_test[1]))

    X_test = getNormal(np.matrix(X_test))
    Y_test = np.matrix(Y_test)
    errorRate = getError(X_test,Y_test,getW)
    print("ErrorRate",errorRate)
    showSample(X_Normal[0:trainingNum], X_Normal[trainingNum:2 * trainingNum], getW, 0, 1)

    return [getW,errorRate]

#使用UCI数据进行测试
def useUCIData(trainingRate=0.6,select="GD"):
    if select=="GD":
        print("\n使用UCI数据,且为梯度上升法")
    else:
        print("\n使用UCI数据,且为牛顿法")
    # 算法中定义的常量
    alpha = 1e-2
    Lambda = 0
    epsilon = 1e-4
    trainNum = int(trainingRate*50)

    sample_train = getSampleFromUCI(0,trainNum)
    X = sample_train[0]
    Y = sample_train[1]
    sample_test = getSampleFromUCI(trainNum,50)
    X_test = sample_test[0]
    Y_test = sample_test[1]
    if select=="GD":
        get = caculateW_GD(np.matrix(X),np.matrix(Y),epsilon,alpha,Lambda)
        print("w:\n",get[0])
        print("ErrorRate:",getError(np.matrix(X_test),np.matrix(Y_test),get[0]))
    else:
        X_Normal = getNormal(np.matrix(X))
        getW = caculateW_NT(np.matrix(X_Normal), np.matrix(Y), Lambda=Lambda)
        print("w:\n",getW)
        X_test = getNormal(np.matrix(X_test))
        Y_test = np.matrix(Y_test)
        errorRate = getError(X_test, Y_test, getW)
        print("ErrorRate:", errorRate)

# 计算p与error的关系
def correlatePWithError(select="GD"):
    m = [0]*11
    x = [0]*11
    for i in range(11):
        if select == "GD":
            m[i] = useMyOwnData(select="GD",p=i/10,printMsg=1)[1]
        else:
            m[i] = useMyOwnData(select="NT",p=i/10,printMsg=1)[1]
        x[i] = i/10
    plt.plot(x,m)
    plt.show()



# 下面代码执行相应功能
#useUCIData(select="GD")
#useMyOwnData(select="GD",p=0.1)
correlatePWithError(select="NT")