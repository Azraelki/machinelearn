import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import struct
from NeuralNetMLP import NeuralNetMLP

def load_mnist(path,kind='train'):
    '''
    从指定路径加载MNIST数据
    :param path:
    :param kind:
    :return:
    '''
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte'%kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte'%kind)

    with open(labels_path,'rb') as lbpath:
        magic,n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)

    with open(images_path,'rb') as imgpath:
        magic,num,rows,cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels),784)
        images = ((images/255.)-0.5)*2 # 将灰度值缩放到[-1,1]上，有利于梯度下降算法

    return images,labels

#
def function1():
    X_train,y_train = load_mnist("../../../machinelearndata/chapter12/",'train')
    print('rows: %d,columns:%d'%(X_train.shape[0],X_train.shape[1]))
    X_test, y_test = load_mnist("../../../machinelearndata/chapter12/", 't10k')
    print('rows: %d,columns:%d' % (X_test.shape[0], X_test.shape[1]))


    # 展示0-9手写字图形，图形数据为28*28像素点
    fig,ax = plt.subplots(nrows=2,ncols=5,sharex=True,sharey=True)

    ax = ax.flatten()
    for i in range(10):
        img = X_train[y_train==i][0].reshape(28,28)
        ax[i].imshow(img,cmap='Greys')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

    # 展示同一个数字手写图形的差异
    fig,ax = plt.subplots(nrows=5,ncols=5,sharey=True,sharex=True)
    ax = ax.flatten()
    for i in range(25):
        img = X_train[y_train==7][i].reshape(28,28)
        ax[i].imshow(img,cmap='Greys')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

    # 使用np将数据压缩储存到文件中
    np.savez_compressed("mnist_scaled.npz",X_train=X_train,y_train=y_train,
                        X_test=X_test,y_test=y_test)

    mnist = np.load("mnist_scaled.npz")
    X_train,y_train,X_test,y_test = [mnist[f] for f in mnist.files]
    print(X_train.shape)

# 使用自己实现的神经网络训练样本
def function2():
    X_train, y_train = load_mnist("../../../machinelearndata/chapter12/", 'train')
    X_test, y_test = load_mnist("../../../machinelearndata/chapter12/", 't10k')

    nn = NeuralNetMLP(n_hidden=100,
                      l2=0.01,
                      epochs=200,
                      eta=0.0005,
                      minibatch_size=100,
                      shuffle=True,
                      seed=1)

    nn.fit(X_train=X_train[:55000],y_train=y_train[:55000],
           X_valid=X_train[55000:],y_valid=y_train[55000:])

    # 绘制 迭代次数-成本函数 图线
    plt.plot(range(nn.epochs),nn.eval_['cost'])
    plt.xlabel("epochs")
    plt.ylabel("cost")
    plt.show()


if __name__ == '__main__':
    # function1()
    function2()

