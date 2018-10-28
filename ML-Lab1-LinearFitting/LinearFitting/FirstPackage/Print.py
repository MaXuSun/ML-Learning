import CreateSample as cs
import GetW as gw
import Utils as us
import math
import matplotlib.pyplot as plt
import numpy as np
# 该模块可以自行设置任意变量运行查看拟合函数和得到Erms值

n = 10   # 设置样本个数
m = 9    # 设置 m 数值
Lambda = 0   # 加入正则项后的Lambda取值
epsilon = 1e-6   # 梯度法的精度(包括梯度下降法和共轭梯度法)
alpha = 0.01      # 梯度下降法的步长
algorithm = "A"    # 根据输入 A 或者 CG 或者 GD 选择不同算法

# 下面两句用来产生数据
a = cs.CreateSample(n)
a.createData()

# 使用Utils工具，得到 X 矩阵,t 向量
utils  = us.Utils(m, a.x, a.y, Lambda)

# 计算得到 W ,可通过修改getW的使用方法，选择使用
# 解析解方法         getW.getW_A
# 还是梯度下降法     getW.getW_GD
# 还是共轭梯度法求解   getW.CG
getW = gw.GetW(utils.X,utils.t,utils.Lambda)
if algorithm == "A":
   w = getW.getW_A()
elif algorithm == "CG":
   w = getW.getW_CG(epsilon)
elif algorithm == "GD":
   w = getW.getW_GD(epsilon, alpha)

# 计算 ERMS
ERMS = utils.calculateE(w)
print("Erms",ERMS)

# 下面几句用来生成拟合的曲线
x = np.linspace(0,1,1000)
y = 0*x
for i in range(np.shape(w)[0]):
   y += pow(x,i)*w[i,0]
print("w",w)
plt.title("Curve Fitting")
plt.plot(x,y,label='M ='+str(m)+","+algorithm+" algorithm")

# 画原函数曲线
xO = np.linspace(0,1,1000)
yO = np.sin(2*math.pi*xO)
plt.plot(xO,yO,color="black",label="y=sin(2PI*x)")
plt.scatter(a.x,a.y)         # 用来 生成描绘生成的数据样本

plt.legend()
plt.show()