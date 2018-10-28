import math
import numpy as np

# 该部分计算根据解析解得出来得的结果

class AnalyticalSolution(object):
    M = 0         # 记录M的取值
    X = []        # 记录得到的X矩阵
    x = []        # 记录得到的x向量
    t = []        # 记录得到的t向量
    w = []        # 记录得到的w向量
    Lambda = 0
    def __init__(self,M,x,y,Lambda):
        self.M = M
        self.x = x
        self.t = np.mat(y)
        self.Lambda = Lambda
        self.generateX()
        self.generateT()
        # self.generateW()
    def generateX(self):     # 生成X矩阵的方法
        for i in self.x:
            j = 0
            row = [];
            while j <= self.M:
                row.append(math.pow(i, j))
                j = j + 1
            if (i == self.x[0]):
                self.X = np.mat(row)
            else:
                self.X = np.row_stack((self.X, row))
    def generateT(self):      # 生成T向量的方法
        self.t  = np.transpose(self.t)
