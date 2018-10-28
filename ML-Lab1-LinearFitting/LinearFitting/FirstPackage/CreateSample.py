import numpy as np
import math
# 该部分生成实验的样本数据

class CreateSample(object):
    x = []   # 自变量数组x
    y = []   # 生成的因变量数组y
    sampleNum = 10  # 生成数据的个数
    mu = 0          # 正态分布的mu
    sigma = 0.1     # 正态分布的sigma
    seed = 3        # 设置的随机数种子
    def __init__(self,sampleNum):      # 参数为样本数量
        self.sampleNum = sampleNum
    def createData(self):
        np.random.seed(self.seed)
        self.x = np.linspace(1.0/self.sampleNum,1,self.sampleNum)
        self.y = np.sin(2.0*np.pi*self.x)
        for i in range(self.x.size):
            self.y[i]+=np.random.normal(self.mu,self.sigma)
    def setNormal(self,mu,sigma):       # 可设置正态分布的 mu 和 sigma
        self.mu = mu
        self.sigma = sigma
    def setSeed(self,seed):             # 可设置随机数种子
        self.seed = seed
