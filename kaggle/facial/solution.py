'''
kaggle: facial keypoints detection

'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# 加载数据

def load_data():
    train = pd.read_csv('../../../machinelearndata/kaggle/facial/training.csv')
    test = pd.read_csv('../../../machinelearndata/kaggle/facial/test.csv')

    # 把图片的数据信息转换为数组
    train['Image'] = train['Image'].apply(lambda im:np.fromstring(im,sep=' '))
    test['Image'] = test['Image'].apply(lambda im:np.fromstring(im,sep=' '))

    # 打印每列的值数量
    print('train data count:\n',train.count())
    print("test data count:\n",test.count())

    # 丢弃缺失数据的行
    train = train.dropna()
    test = test.dropna()

    # 获取数据，并将图片数据RGB值转换到（0,1）上
    X_train = np.vstack(train['Image'].values)/255
    X_train = X_train.astype(np.float32)
    X_test = np.vstack(test['Image'].values)/255
    X_test = X_test.astype(np.float32)

    # 获取目标值
    y = train.iloc[:,:-1].values
    y = (y-48)/48 # 将值转换到（-1,1）上
    y = y.astype(np.float32)

    X_train,y = shuffle(X_train,y,random_state=123)
    # 返回读取的数据(训练集，训练目标，测试集)
    return X_train,y,X_test

X_train,y_train,X_test = load_data()
print("X_train shape=={};y_train shape=={}".format(X_train.shape,y_train.shape))
print("X_test shape=={}".format(X_test.shape))

def batch_generator(X,y,batch_size=64):
    X_copy = np.copy(X)
    y_copy = np.copy(y)

    for i in range(0,X.shape[0],batch_size):
        yield (X_copy[i:batch_size,:],y_copy[i:batch_size,:])

# 训练一个简单的单层网络
def single_layer_net():
    # 构造计算图
    g = tf.Graph()
    with g.as_default():
        # 输入占位符
        tf_x = tf.placeholder(tf.float32,shape=(None,9216),name='tf_x')
        tf_y = tf.placeholder(tf.float32,shape=(None,30),name='tf_y')
        # 隐藏层-1
        h1 = tf.layers.dense(inputs=tf_x,units=100,activation=tf.nn.relu)
        # 输出层
        h2 = tf.layers.dense(inputs=h1,units=30,activation=tf.nn.tanh)

        # 定义成本函数
        cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf_y,predictions=h2),name='mse_cost')

        # 预测
        prediction = {
            'keypoint':tf.add(tf.multiply(h2,tf.constant(48,dtype=tf.float32)),tf.constant(48,dtype=tf.float32))
        }
        # 定义优化器
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999)
        train_op = optimizer.minimize(loss=cost)
        init_op = tf.global_variables_initializer()

    # 创建会话
    with tf.Session(graph=g) as sess:
        tf.random.set_random_seed(123)
        sess.run(init_op)
        costs = []
        # 迭代
        for epoch in range(400):
            generator = batch_generator(X_train,y_train)
            avg_cost = 0
            for batch_x,batch_y in generator:
                feed = {
                    tf_x:batch_x,
                    tf_y:batch_y
                }
                _,c = sess.run([train_op,cost],feed_dict=feed)
                avg_cost += c
            costs.append(avg_cost)
            print('epoch-{} cost:{}'.format(epoch+1,avg_cost))


single_layer_net()







