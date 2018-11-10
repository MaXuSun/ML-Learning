import numpy as np
import matplotlib.pyplot as plt

# 生成不同高斯分布的二维特征数据集


def createSample(mu, sigma, p=0, sampleNum=50):
    # 二维正态分布
    # np.random.seed(2)
    Sigma = [[sigma[0] ** 2, p * sigma[0] * sigma[1]],
             [p * sigma[0] * sigma[1], sigma[1] ** 2]]  # 协方差矩阵
    x, y = np.random.multivariate_normal(mu, Sigma, sampleNum).T

    return np.vstack((x, y)).T

# 从fileName文件中读取数据


def loadDatSet(fileName, start=0, end=7):
    data = []
    fr = open(fileName)
    for line in fr.readlines():
        temp = line.split()[0:7]
        data.append(list(map(float, temp)))
    return np.array(data)


def disTwoVector(vec1, vec2):
    return np.sqrt(np.sum(np.power(vec1-vec2, 2)))

# 初始化质心


def initCenter(data, k):
    # print("data",data)
    m, n = np.shape(data)
    center = np.matrix(np.zeros((k, n)))           # 生成 k x n的质点
    # 接下来按行生成数据，取没列的最大值和最小值，然后随机生成一列随机数在该范围内
    rangeCol = np.array(np.max(data, 0)-np.min(data, 0))
    for i in range(n):
        center[:, i] = (rangeCol[i]*np.matrix(np.random.rand(k, 1)))[:, 0]
    return center


# 使用k-means方法进行分类
# 参数：
# data  : 数据集
# k     : 分的类别个数
def kMeans(data, k):
    m, n = np.shape(data)
    # 初始化每个点的最近质点及到该质点的距离  [质点，距离]
    point2centerData = np.matrix(np.zeros((m, 2)))
    center = initCenter(data, k)                     # 初始化k个质点中心
    Iteration = True                                # 控制是否继续迭代

    while Iteration:                                # 只要空间中有点的最近质点发生变化就继续迭代
        Iteration = False

        for i in range(m):                          # 对空间中所有点进行遍历，找每个点的最近质点并更新该点到质点的数据

            minDist = np.inf                        # 找到离得最近的质点
            minIndex = -1
            for j in range(k):
                tempDist = disTwoVector(center[j, :], data[i, :])
                if tempDist < minDist:
                    minDist = tempDist
                    minIndex = j

            if point2centerData[i, 0] != minIndex:         # 只要空间中有一个点的最近质点发生了变化，就需要继续迭代
                Iteration = True

            # 更新每个点的最近质点以及离最近质点的距离
            point2centerData[i, :] = minIndex, minDist**2

        # print(center)                                    # 打印这几个质点

        for cen in range(k):                             # 更新每个质点位置
            ptsInClust = data[np.nonzero(point2centerData[:, 0].A == cen)[
                0]]                                      # 获取最近质点为 k质点的所有点(通过np.nonzero函数找到
                                                         # 所有分的簇为cen的样本点)
            if(ptsInClust.size == 0):                    # 如果这次这个族没有样本点就直接跳过
                continue
            temp = np.mean(ptsInClust, axis=0)
            center[cen, :] = temp                        # 更新 质点位置

    return center, point2centerData

# 计算高斯分布概率
# x   : 某个样本
# s   : 某个高斯分布的 sigma
# u   : 某个高斯分布的 mu
def Gs(x,s,u):
    x = x.T
    u = u.T
    s = np.matrix(s)
    x = np.matrix(x)
    n,m = np.shape(x)
    sigma = (np.linalg.det(s))**(1/2)
    e = np.ma.exp((-1/2)*((x-u).T*s.I*(x-u)))     # 算指数部分
    s = 1/(((2*np.pi)**(n/2))*sigma)       # 算系数部分
    return e[0,0]*s

# 每个高斯分布下计算 x 发生的概率
# x    : 某个样本
# sigma:所有高斯分布的 sigma 数组
# mu   :所有高斯分布的 mu 数组
# alpha:所有高斯分布的 alpha 数组

def px(x,sigma,mu,alpha):
    p = []
    k = alpha.__len__()
    for i in range(k):
        p.append(alpha[i]*Gs(x,sigma[i],mu[i]))
    return p

def max_like(data,sigma,mu,alpha):
    m,n = np.shape(data)
    ll = 0
    for j in range(m):
        px_j = px(data[j], sigma, mu, alpha)  # x_j 由每个高斯分布产生的概率
        ll += np.log(np.sum(px_j))
    return ll

# EM算法进行分类
# data : 数据集 X
# k    : 质点划分个数
# loop : 迭代次数
# s    : 初始sigma取值
def EM(data,k,loop = 100,s=0.1):
    m,n = np.shape(data)
    alpha = np.array([1/k]*k)                                   # 初始化 alpha     (1*k)
    sigma = np.array([s*np.eye(n)]*k)                         # 初始化 sigma     k 个 s 每个为`(n*n)
    mu = np.matrix(initCenter(data,k))                          # 初始化 mu        (m*n)
    data = np.matrix(data)
    r = np.matrix(np.zeros((m,k)))                              # 用来记录r_ji x_j 数据是由第i个高斯分布产生的概率 (m*k)
    maxLike = []

    ite = 0
    while ite < loop:
        first = max_like(data,sigma,mu,alpha)
        maxLike.append(first)
        # 这几步是计算后验概率
        for j in range(m):
            px_j = px(data[j],sigma,mu,alpha)                    # px_j 由每个高斯分布产生的概率
            r[j] = px_j/np.sum(px_j)                             # 用来更新 r 矩阵

        # 这几步更新参数
        for i in range(k):                                       # 更新 mu
            a = 0
            b = 0
            for j in range(m):
                a += r[j,i]*data[j]
                b += r[j,i]
            mu[i] = a/b

        for i in range(k):                                       # 更新sigma
            a = 0
            b = 0
            for j in range(m):
                a += r[j,i]*((data[j]-mu[i]).T*(data[j]-mu[i]))
                b += r[j,i]
            sigma[i] = a/b

        for i in range(k):                                       # 更新alpha
            a = 0
            for j in range(m):
                a += r[j,i]
            alpha[i] = a/m
        ite+=1
    # 根据结果对点分类
    max_r = np.argmax(r,1).T.A[0]
    dict = {}
    for i in range(k):
        dict[i] = []
    for i in range(len(max_r)):
        dict[max_r[i]].append(data[i].A[0])
    print("mu",mu)
    print("sigma",sigma)
    return dict,maxLike

# 使用UCI数据进行测试，UCI数据地址为：
# https://archive.ics.uci.edu/ml/datasets/seeds
# 其中类别3个类别，共有210个数据，每个数据有7个特征
# 已被缓存在本地的uicdata文件中

def useUciData(loop=5,s=0.01):
    print("使用UCI数据进行测试:")
    get = loadDatSet("ucidata")
    print("使用 K-Means 得到的结果：")
    cent, assum = kMeans(get, 3)
    print("Center\n", cent)
    print("使用 EM 得到的结果：")
    EM(get,3,loop=loop,s=s)

# 使用自己的数据进行测试
def mydataToKmeans(num=100):
    print("使用自己的数据测试 K-Means:")
    mu1 = [0.3, 0.3]  # 平均值
    sigma1 = [0.1, 0.1]  # 方差

    mu2 = [0.4, 0.7]
    sigma2 = [0.1, 0.1]

    mu3 = [0.7, 0.2]
    sigma3 = [0.1, 0.1]

    get1 = createSample(mu1, sigma1, sampleNum=num)
    get2 = createSample(mu2, sigma2, sampleNum=num)
    get3 = createSample(mu3, sigma3, sampleNum=num)

    plt.subplot(2, 2, 1)     # 几行几列的第几个
    plt.title("Source data set")
    plt.plot(get1[:, 0], get1[:, 1], "+", color="red")
    plt.plot(get2[:, 0], get2[:, 1], "x", color="blue")
    plt.plot(get3[:, 0], get3[:, 1], ".", color="green")

    plt.subplot(2, 2, 2)
    plt.title("K-Means get")
    plt.plot(get1[:, 0], get1[:, 1], "+", color="red")
    plt.plot(get2[:, 0], get2[:, 1], "x", color="blue")
    plt.plot(get3[:, 0], get3[:, 1], ".", color="green")
    get = np.vstack((get1, get2, get3))
    cent, assum = kMeans(get, 3)
    print("Center\n", cent)
    plt.plot(cent[:, 0], cent[:, 1], "o", color="black")
    # plt.show()

def mydataToEm(num=100):
    print("使用自己的数据测试 EM 算法:")
    mu1 = [0.3, 0.3]  # 平均值
    sigma1 = [0.1, 0.1]  # 方差

    mu2 = [0.4, 0.7]
    sigma2 = [0.1, 0.1]

    mu3 = [0.7, 0.2]
    sigma3 = [0.1, 0.1]

    get1 = createSample(mu1, sigma1, sampleNum=num)
    get2 = createSample(mu2, sigma2, sampleNum=num)
    get3 = createSample(mu3, sigma3, sampleNum=num)

    plt.subplot(2, 3, 4)  # 几行几列的第几个
    plt.title("Source data set")
    plt.plot(get1[:, 0], get1[:, 1], "+", color="red")
    plt.plot(get2[:, 0], get2[:, 1], "x", color="blue")
    plt.plot(get3[:, 0], get3[:, 1], ".", color="green")
    get = np.vstack((get1, get2, get3))
    dict,maxLike = EM(get,3)
    signals = ["+","x","."]
    colors = ["blue","red","green"]
    plt.subplot(2,3,5)
    plt.title("EM get")
    for i in range(len(dict)):
        x = np.array(dict[i])
        a1 = x[:,0]
        a2 = x[:,1]
        plt.plot(a1,a2,signals[i],color = colors[i])
    plt.subplot(2,3,6)
    plt.plot(range(maxLike.__len__()),maxLike)
    plt.show()

if __name__ == '__main__':
    plt.figure(figsize=[10,10])
    mydataToKmeans()
    mydataToEm()
    # useUciData(loop=20,s=9)