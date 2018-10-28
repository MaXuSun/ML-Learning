import numpy as np
import math
# 该部分为计算w的主要函数部分

class GetW(object):
    X = 0
    t = 0
    w = 0
    Lambda = 0
    num = 0
    def __init__(self,X,t,Lambda):
        self.X = X
        self.t = t
        self.Lambda = Lambda

    def haveSolution(self):         # 判断是否有解，有解返回1，没有返回0
        XshapeRow = np.shape(self.X)[1]  # 计算矩阵X的列数
        temp1 = np.dot(np.transpose(self.X), self.X) + np.eye(XshapeRow) * self.Lambda
        temp2 = np.dot(np.transpose(self.X),self.t)
        temp3 = np.column_stack((temp1,temp2))
        if(np.linalg.matrix_rank(temp1) == np.linalg.matrix_rank(temp3)):
            return 1
        else:
            return 0

    def getW_A(self):                                 # 使用解析解计算
        XshapeRow = np.shape(self.X)[1]               # 计算矩阵X的列数
        temp1 = np.dot(np.transpose(self.X),self.X)+np.eye(XshapeRow)*self.Lambda    # 计算X^T*X+Lambda*I
        temp2 = np.dot(np.linalg.inv(temp1),np.transpose(self.X))       # 计算(X^T*X+Lambda*I)^-1 * X^T
        self.w = np.dot(temp2,self.t)                                   # 计算(X^T*X+Lambda*I)^-1 * X^T*T
        return self.w

    def getW_GD(self,epsilon,alpha):                 #使用梯度下降法计算
        w = np.zeros((np.shape(self.X)[1],1))
        EW_G = lambda W:self.X.T*(self.X*W-self.t)+self.Lambda*W    # 对EW的求导项得到的结果
        self.num = 0                                                # 记录迭代次数
        while self.num < 100000:
            temp = EW_G(w)*alpha                                    # 记录下降的距离
            if(math.fabs(temp.max())< epsilon):                     # 判断是否达到相应精度
                break
            else:
                w = w - temp                                        # 更新 w
            self.num+=1
        self.w = w
        return self.w


    def getW_CG(self,epsilon):                            # 使用共轭梯度法计算
        # 初始化一些数据
        w = np.zeros((self.X.shape[1], 1))                          # 初始化 w0
        I = np.eye(self.X.shape[1])
        A = (self.X.T * self.X) + self.Lambda * I                   # 初始化矩阵 A
        b = self.X.T * self.t                                       # 初始化向量 b
        r = b-A*w                                                   # 初始化 r0
        p = r.copy()                                                # 初始化 p0
        rsold = (r.T*r)[0,0]                                        # 旧的 r^T*r 值

        k = 0
        while k < np.shape(b)[0]:
            pAp = p.T*A*p
            alpha = rsold /pAp[0, 0]                               # 求出 alpha
            w =w + alpha * p                                       # 求出 w(k+1)并更新 w
            r = r-alpha*A*p                                        # 出 r(k+1)并更新 r
            rsnew = (r.T*r)[0,0]
            if(rsnew<epsilon):  # 根据传入的精度判断是否终止循环
                break
            else:
                p=r+(rsnew/rsold)*p                                # 更新p
                rsold = rsnew
                k +=1
        self.w = w
        return self.w
