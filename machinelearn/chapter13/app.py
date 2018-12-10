import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import struct
from TFLinreg import TFLinreg


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

# tensorflow 实现 z=w*x+b
def function1():
    # 创建graph对象
    g = tf.Graph()
    with g.as_default():
        x = tf.placeholder(dtype=tf.float32,shape=(None),name='x')

        w = tf.Variable(2.0,name='weight')
        b = tf.Variable(0.7,name='bias')

        z = w*x + b
        init = tf.global_variables_initializer()

    # 创建session对象并传入graph
    with tf.Session(graph=g) as sess:
        # 初始化参数 w和b
        sess.run(init)
        # 计算z
        for t in [1.0,0.6,-1.8]:
            print("x=%4.1f --> z=%4.1f"%(t,sess.run(z,feed_dict={x:t})))

# 使用数组形式
def function2():
    g = tf.Graph()
    with g.as_default():
        x = tf.placeholder(dtype=tf.float32,shape=(None,2,3),name='input_x')
        x2 = tf.reshape(x,shape=(-1,6),name='x2')
        # 计算每列的和
        xsum = tf.reduce_sum(x2,axis=0,name='col_sum')

        # 计算每列的均值
        xmean = tf.reduce_mean(x2,axis=0,name='col_mean')

    with tf.Session(graph=g) as sess:
        x_array = np.arange(18).reshape(3,2,3)
        print("input shape: ",x_array.shape)
        print("column sums:\n",sess.run(x2,feed_dict={x:x_array}))
        print("column sums:\n",sess.run(xsum,feed_dict={x:x_array}))
        print("column means:\n",sess.run(xmean,feed_dict={x:x_array}))

# 使用自己实现的TFLinreg训练模型
def train_linreg(sess,model,X_train,y_train,num_epochs):
    # 初始化模型定义的所有参数 w 和 b
    sess.run(model.init_op)

    training_costs = []
    for i in range(num_epochs):
        _, cost = sess.run([model.optimizer,model.mean_cost],
                           feed_dict={model.X:X_train,model.y:y_train})
        training_costs.append(cost)
    return training_costs
# 使用自己实现的TFLinreg进行预测
def predict_linreg(sess,model,X_test):
    y_pred = sess.run(model.z_net,feed_dict={model.X:X_test})
    return y_pred

def function3():
    X_train = np.arange(10).reshape((10,1))
    y_train = np.arange(1,11)

    lrmodel = TFLinreg(x_dim=X_train.shape[1],learning_rate=0.01)
    sess = tf.Session(graph=lrmodel.g)
    train_costs = train_linreg(sess,lrmodel,X_train,y_train,10)

    # 绘制成本函数随迭代次数的改变
    plt.plot(range(1,len(train_costs)+1),train_costs)
    plt.tight_layout()
    plt.xlabel("epochs")
    plt.ylabel("training cost")
    plt.show()

    # 绘制样本点和拟合的直线
    plt.scatter(X_train,y_train,marker='s',s=50,label='training data')
    plt.plot(range(X_train.shape[0]),predict_linreg(sess,lrmodel,X_train),
             color='gray',marker='o',markersize=6,lw=3,label='linreg model')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.show()


# 加载数据集，并使用layer实现手写字识别
def function4():
    X_train, y_train = load_mnist("../../../machinelearndata/chapter12/", 'train')
    print('rows: %d,columns:%d' % (X_train.shape[0], X_train.shape[1]))
    X_test, y_test = load_mnist("../../../machinelearndata/chapter12/", 't10k')
    print('rows: %d,columns:%d' % (X_test.shape[0], X_test.shape[1]))


if __name__ == '__main__':
    # function1()
    # function2()
    # function3()
    function4()
