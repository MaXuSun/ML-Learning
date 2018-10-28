import CreateSample as cs
import GetW as gw
import Utils as us
import math
import matplotlib.pyplot as plt
import numpy as np
# 改模块运行会比较不同 样本数量 对于拟合的影响,先后显示两个图像,分别是不同样本数拟合的所有图像与不同M值对应得Erms(训练集和测试集)图像
# 通过修改循环的初始值和叠加值及终止值控制 样本数量 的循环
start = 5
feed = 5
end = 150

m = 9    # 设置 m 数值
Lambda = 0  # 加入正则项后的Lambda取值
epsilon = 0.001   # 梯度法的精度(包括梯度下降法和共轭梯度法)
alpha = 0.002      # 梯度下降法的步长
algorithm = "CG"    # 根据输入 A 或者 CG 或者 GD 选择不同算法
ErmsTest = []
ErmsTraining = []
N = []

def mytest(a):
    # 得到 X 矩阵 ， t 向量
    utils = us.Utils(m,a.x,a.y,Lambda)
    # 选择算法
    getW = gw.GetW(utils.X,utils.t,utils.Lambda)
    if algorithm == "A":
        w = getW.getW_A()
    elif algorithm =="CG":
        w = getW.getW_CG(epsilon)
    elif algorithm == "GD":
        w = getW.getW_GD(epsilon,alpha)
    # 计算 Erms
    ermsTest = utils.calculateE(w)
    ErmsTest.append(ermsTest)
    ermsTraining = utils.caculateTraining(w)
    ErmsTraining.append(ermsTraining)
    # 下面几句用来生成拟合的曲线
    x = np.linspace(0,1,1000)
    y = 0*x
    for i in range(np.shape(w)[0]):
        y += pow(x,i)*w[i,0]
    plt.title("Curve Fitting")
    plt.plot(x,y,label='SampleNum ='+str(np.shape(a.x)[0]))
    print('M =' + str(m) + " ,ErmsTest:", ermsTest)
    print('M =' + str(m) + " ,ErmsTraining:", ermsTraining)
    # /plt.scatter(a.x,a.y)


def sim(start,feet,end):
    n = start
    # 下面两句产生数据样本
    while n <=end:
        N.append(n)
        a = cs.CreateSample(n)
        a.createData()
        mytest(a)
        n += feet

sim(start,feed,end)

# 画原函数曲线
xO = np.linspace(0,1,1000)
yO = np.sin(2*math.pi*xO)
plt.plot(xO,yO,color="black",label="y=sin(2PI*x)")
# print(Erms)
plt.legend()
plt.show()
# 下面是画样本数量与Erms的关系
plt.title("Erms for different Sample Number")
plt.plot(N,ErmsTest,color="red",label="Test")
# plt.plot(N,ErmsTraining,color="red",label="Training")
plt.legend()
plt.show()

