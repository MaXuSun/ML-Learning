import math
import numpy as np
import CreateSample as cs

# 该部分用来根据传入的 M,x,y,和Lambda来计算需要的矩阵 X 和向量 t
# 及计算Erms

class Utils(object):
    M = 0         # 记录M的取值
    X = []        # 记录得到的X矩阵
    t = []        # 记录得到的t向量
    Lambda = 0
    ERMS = 0
    def __init__(self,M,x,y,Lambda):   # 参数为 M,x,y,Lambda
        self.M = M
        self.t = np.mat(y)
        self.Lambda = Lambda
        self.X = self.generateX(x)
        self.t = self.generateT(y)
    def generateX(self,x):     # 生成X矩阵的方法
        X = 0
        for i in x:
            j = 0
            row = [];
            while j <= self.M:
                row.append(math.pow(i, j))
                j = j + 1
            if (i == x[0]):
                X = np.mat(row)
            else:
                X = np.row_stack((X, row))
        return X
    def generateT(self,t):      # 生成T向量的方法
        return np.transpose(np.mat(t))
    def calculateE(self,w):                          # 计算测试集的Erms
        a = cs.CreateSample(15)
        a.setSeed(346)
        a.createData()
        X = self.generateX(a.x)
        t = self.generateT(a.y)
        EW = lambda W:0.5*((X*W-t).T*(X*W-t))
        N = np.shape(X)[0]
        self.ERMS = math.pow(2*EW(w)/N,0.5)
        return self.ERMS
    def caculateTraining(self,w):                      # 计算训练集的Erms
        EW = lambda W:0.5*((self.X*W-self.t).T*(self.X*W-self.t))
        N = np.shape(self.X)[0]
        self.ERMS = math.pow(2*EW(w)/N,0.5)
        return self.ERMS