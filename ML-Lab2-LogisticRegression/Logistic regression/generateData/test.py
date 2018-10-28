import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
x = np.matrix([[1],[2],[3],[4]])
x[2][0] = 5
print(x)

df = pd.read_csv('../ucidata/iris.data') # 加载Iris数据集作为DataFrame对象
X = df.iloc[:, [0, 2]].values # 取出2个特征，并把它们用Numpy数组表示

plt.scatter(X[:50, 0], X[:50, 1],color='red', marker='o', label='setosa') # 前50个样本的散点图
plt.scatter(X[50:100, 0], X[50:100, 1],color='blue', marker='x', label='versicolor') # 中间50个样本的散点图
plt.scatter(X[100:, 0], X[100:, 1],color='green', marker='+', label='Virginica') # 后50个样本的散点图
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc=2) # 把说明放在左上角，具体请参考官方文档
plt.show()
# x = np.ones((np.shape(X)[0],1))
# sample = np.hstack((x,X))
# y1 = sample[:40]
# y2 = sample[50:90]
# print(y1)
# print(y2)
# print(np.shape(y1))
# print(np.shape(y2))
# print(np.shape(np.vstack((y1,y2))))

# X = np.mat([[1],[2],[3]])
# Y = [2,3,4]
# print(np.diag(Y))
# print(np.ones(3))
# print(np.diag(np.diag(X*np.ones(3))))




## 使用拟牛顿法计算 W (并且在找相识Hessian时用BFGS算法)
# def caculateW_BFGS(X,Y,epsilon,Lambda):
#     m = np.shape(X)
#     Lambda  = 0
#     Gk = lambda W: X.T * (Y - np.exp(X * W) / (1 + np.exp(X * W))) + Lambda * W
#     w = np.zeros((m[1],1))
#     D = np.eye(m[1])    #初始化D0
#     k = 0                          #初始化k
#     d = -D * Gk(w)
#     s = findFeet(X,Y,w,d)*d
#     w1 = w + s
#     while math.fabs(Gk(w1).max()) > epsilon:
#         y = Gk(w1)-Gk(w)
#         D = (np.eye(m[1])-(s*y.T)/(y.T*s))*D*(np.eye(m[1])-(y*s.T)/(y.T*s))+(s*s.T)/(y.T*s)
#         w = w1
#         d = -D *Gk(w)
#         s = findFeet(X,Y,w,d)*d
#         w1 = w+s
#     return w1
#
# def findFeet(X,Y,w,dk):          # BFGS线性搜索步长
#     Lw = lambda W: Y.T * X * W - np.ones((1, np.shape(X)[0])) * np.log(1 + np.exp(X * W))
#     Gk = lambda W: X.T * (Y - np.exp(X * W) / (1 + np.exp(X * W))) + Lambda * W  # 梯度计算公式
#     rho = 0.55
#     deta = 0.4
#     alpha = 0
#     alphaK = 0
#     while alpha < 20:
#         newf= Lw(w+(rho**alpha)*dk)
#         old = Lw(w)
#         if(newf < old+deta*(rho**alpha)*Gk(w).T*dk):
#             alphaK = alpha;
#             break
#         alpha += 1
#     return rho**alphaK


def caculateW_NT(X,Y,epsilon):
    Xw = lambda W: np.exp(X*W)/(1+np.exp(X*W))
    m = np.shape(X)
    w = np.zeros((m[1],1))    # 初始化 w
    P = np.diag(np.diag(Xw(w)*np.ones(m[0])))    # 生成以Xw(w)为对角线的矩阵
    A = P.T*P-P            # 计算A
    H = X.T*A*X             # 计算H'
    U = X.T*(Y-Xw(w))         # 计算U
    deta = np.linalg.inv(H)*U   # 计算Xnew - Xold 即 (H')^-1*U
    i = 0
    while np.fabs(deta).max() > epsilon:
        w = w-deta               # 更新w
        P = np.diag(np.diag(Xw(w) * np.ones(m[0])))
        A = P-P.T*P             # 更新A
        H = X.T*A*X            # 更新H'
        U = X.T*(Y-Xw(w))        # 更新U
        deta = np.linalg.inv(H)*U # 更新 Xnew - Xold
        i+=1
    return w





def caculateW_NT(X,Y,epsilon,Num):
    XMin = X.min(0)
    XMin[0,0] = 0
    XMax = X.max(0)
    X = (X-XMin)/(XMax-XMin)
    print("X", X)
    print("Y", Y)
    Xw = lambda W: np.exp(X*W)/(1+np.exp(X*W))
    m = np.shape(X)
    w = np.ones((m[1],1))    # 初始化 w
    print("w",w)
    print("Xw",Xw(w))
    P = np.diag(np.diag(Xw(w)*np.ones(m[0])))
    print("P",P)
    A = P.T*P-P            # 计算A
    print("A",A)
    H = X.T*A*X             # 计算H
    print("H",H)
    U = X.T*(Y-Xw(w))         # 计算U
    print("U",U)
    deta = np.linalg.pinv(H)*U   # 计算Xnew - Xold 即 (H)^-1*U
    print("deta",deta)
    i = 0
    while i <= Num:
        w = w-deta               # 更新w
        print("w",i,w)
        P = np.diag(np.diag(Xw(w) * np.ones(m[0])))
        print("P", i, P)
        A = P-P.T*P             # 更新A'
        if A.max() == 0:
            return w
        print("A", i, A)
        H = X.T*A*X            # 更新H'
        print("H", i, H)
        U = X.T*(Y-Xw(w))        # 更新U
        print("U", i, U)
        deta = np.linalg.pinv(H)*U # 更新 Xnew - Xold
        print("deta", i, deta)
        i+=1
    return w