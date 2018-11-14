import numpy as np
import matplotlib.pyplot as plt
import os
import struct
import  math
from mpl_toolkits.mplot3d import Axes3D

# pca算法
# 参数：
# X   ：数据集     m x n 维度的数据集
# d   : 低维度空间的维数
def pca(X,d=100):
    mean = np.mean(X,axis=0)                # 求得每一列对应的平均值
    mean_X = X - mean                       # 对数据进行去中心处理
    cov_X = mean_X.T*mean_X                 # 计算协方差矩阵
    vals,vect=np.linalg.eig(cov_X)          # 得到协方差矩阵的特征向量和特征值
    sort = np.argsort(vals)                 # 对特征值进行从小到大排序，得到对应索引
    sort = sort[:-(d+1):-1]                 # 从大到小取出 d 个特征值对应的索引
    v_get = vect[:,sort]                    # 得到排序后的特征值对应的特征向量   (n * d)
    low_X = mean_X*v_get                    # 得到低维度的数据                  (m * d)
    new_X = (low_X*v_get.T)+mean          # 降维之后再重构得到的数据           (m * n)
    return low_X,new_X

# 用于打印3D的点集
def plant3D(X,Y,Z):
    plt.figure(figsize=[10, 5])
    ax = plt.subplot(121,projection='3d')
    plt.title("Front View")
    ax.scatter(X, Y, Z, marker="^")
    ax.view_init(elev=30, azim=-45)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    bx = plt.subplot(122, projection='3d')
    plt.title("Side View")
    bx.scatter(X, Y, Z, marker="^")
    bx.view_init(elev=0, azim=45)
    bx.set_xlabel('X Label')
    bx.set_ylabel('Y Label')
    bx.set_zlabel('Z Label')
    plt.show()

def geneData(num = 1000,size=4,amplitude = 1):
    X = np.random.random(num)*size - size/2
    Y = np.random.random(num)*size - size/2
    rand_Z = np.random.random(num) * amplitude-amplitude/2
    Z = -X + Y + rand_Z
    data = np.vstack((X, Y, Z)).T
    return np.matrix(data)

# 对自己的数据集进行测试
# 参数：
# amplitude  : 生成数据时的抖动幅度
def testMyData(amplitude = 1):
    X = geneData(amplitude=amplitude)
    lowX, nowX = pca(X, 2)
    print(lowX)
    plant3D(X[:,0],X[:,1],X[:,2])
    plt.title("Low dimensional data set")
    plt.plot(lowX[:, 0], lowX[:, 1], ".")
    plt.show()
    plant3D(nowX[:,0],nowX[:,1],nowX[:,2])

# 从文件中加载 mnist 数据
# 参数
# path   : mnist所在的目录
# kind   : 选择读取训练数据或者测试数据
def load_mnist(path,kind='train',train_num=100):
    lables_path = os.path.join(path,'%s-labels.idx1-ubyte'%kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte'%kind)
    with open(lables_path,'rb') as lbpath:
        magic,n = struct.unpack('>II',lbpath.read(8))    # 跳过标签文件的前8字节说明
        labels = np.fromfile(lbpath,dtype=np.uint8)      # 读取标签文件中的标签数据
    with open(images_path,'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))   # 跳过数据文件的前16个字节
        # 读取数据文件的所有数据，并转换为列数为 (num * 784)的数据集
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(num, rows*cols)

    return images[0:train_num,:],labels[0:train_num]


# 显示数字为 num 的图片，显示的个数为 row * col个
# imgs     : 从mnist数据集中读取的数据，规格为 ： m * 784
# num      : 想要输出的图片数字
# row      : 打印的行数
# col      : 打印的列数
def showImgbyNum(imgs,labs,num,row=2,col=5):
    fig, ax = plt.subplots(
        nrows=row,
        ncols=col,
        sharex=True,
        sharey=True,
    )
    ax = ax.flatten()
    for i in range(row*col):
        img = imgs[labs == num][i].reshape(28, 28)
        ax[i].imshow(img)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    # plt.show()

# 打印 0~9 这10个数字，分为 2 行 5列打印
def showOnetoTen(imgs,labs):
    fig, ax = plt.subplots(
        nrows=2,
        ncols=5,
        sharex=True,
        sharey=True,
    )
    ax = ax.flatten()
    for i in range(10):
        img = imgs[labs == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap="Greys")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    # plt.show()

# 所有图片100张图片
def showallImg(imgs,labs):
    fig, ax = plt.subplots(
        nrows=10,
        ncols=10,
        sharex=True,
        sharey=True,
    )
    ax = ax.flatten()
    for i in range(100):
        img = imgs[i].reshape(28, 28)
        ax[i].imshow(img, cmap="Greys")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    # plt.show()

def SNR(X,X_):
    X = np.matrix(X)
    X_ = np.matrix(X_)
    Y = X-X_
    m = np.power(X,2)
    n = np.power(Y,2)
    all = 10*np.log10(np.sum(m,axis=1)/np.sum(n,axis=1))
    return all,np.mean(all)

# 测试mnist数据
# 参数：
# d   : 降到多少维度
def testMnist(d = 1,num=7):
    path = '../mnist'
    train_imgs, train_labs = load_mnist(path=path)
    print(np.shape(train_imgs),type(train_imgs))
    print(np.shape(train_labs),type(train_labs))

    # showImgbyNum(train_imgs,train_labs,num)
    # showOnetoTen(train_imgs,train_labs)
    showallImg(train_imgs,train_labs)
    low_imgs, new_imgs = pca(np.matrix(train_imgs), d=d)
    images = np.array(new_imgs,dtype=np.float)
    # showImgbyNum(images,train_labs,num)
    # showOnetoTen(images,train_labs)
    showallImg(images,train_labs)
    snr,snr_mean = SNR(train_imgs,np.matrix(images))
    print("平均信噪比",snr_mean)
    plt.show()

if __name__ == '__main__':
    testMnist(d=100)
    # testMyData(amplitude=0)