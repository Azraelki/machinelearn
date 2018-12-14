import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.signal
import scipy.misc
import os
import struct

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

# 手工实现卷积计算（向量）
def convld(x,w,p=0,s=1):
    '''卷积
    :param x: 输入向量
    :param w: 滑动的窗口（kernel）
    :param p: 零填充的边距
    :param s: 步幅
    :return: 输出向量
    '''
    w_rot = np.array(w[::-1]) # 翻转kernel
    x_padded = np.array(x)
    if p > 0:
        zero_pad = np.zeros(shape=p)
        x_padded = np.concatenate([zero_pad,x_padded,zero_pad]) # 填充边距
    res = []
    for i in range(0,int(len(x)/s),s):
        res.append(np.sum(x_padded[i:i+w_rot.shape[0]]*w_rot))
    return np.array(res)
# 手工实现卷积计算（二维矩阵）
def conv2d(X,W,p=(0,0),s=(1,1)):
    W_rot = np.array(W)[::-1,::-1] # 两个维度都翻转
    X_orig = np.array(X)
    n1 = X_orig.shape[0] + 2*p[0]
    n2 = X_orig.shape[1] + 2*p[1]
    X_padded = np.zeros(shape=(n1,n2))
    X_padded[p[0]:p[0]+X_orig.shape[0],p[1]:p[1]+X_orig.shape[1]] = X_orig # 填充后的矩阵
    res = []
    for i in range(0,int((X_padded.shape[0]-W_rot.shape[0])/s[0])+1,s[0]):
        res.append([])
        for j in range(0,int((X_padded.shape[1]-W_rot.shape[1])/s[1])+1,s[1]):
            X_sub = X_padded[i:i+W_rot.shape[0],j:j+W_rot.shape[1]]
            res[-1].append(np.sum(X_sub*W_rot))
    return (np.array(res))

def function1():
    # 一维卷积
    x = [1,3,2,4,5,6,1,3]
    w = [1,0,3,1,2]
    print("convld implementation:",convld(x,w,p=2,s=1))
    print("numpy result:", np.convolve(x, w, mode='same'))

    # 二维卷积
    X = [[1, 3, 2, 4], [5, 6, 1, 3], [1, 2, 0, 2], [3, 4, 3, 2]]
    W = [[1, 0, 3], [1, 2, 1], [0, 1, 1]]
    print('Conv2d Implementation:\n',conv2d(X, W, p=(1, 1), s=(1, 1)))
    print("scipy results:",scipy.signal.convolve2d(X,W,mode='same'))

# 使用scipy.misc读取图片
def function2():
    # 读取图片
    img = scipy.misc.imread('./example-image.png',mode='RGB')
    print("image shape:",img.shape)
    print("number of channels:",img.shape[2])
    print("image data type:",img.dtype)


def batch_generator(X,y,batch_size=64,shuffle=False,random_seed=None):
    '''批数据生成器
    :param X:
    :param y:
    :param batch_size:
    :param shuffle:
    :param random_seed:
    :return:
    '''
    idx = np.arange(y.shape[0])
    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]
    for i in range(0,X.shape[0],batch_size):
        yield (X[i:i+batch_size,:],y[i:i+batch_size])
################# 使用卷积神经网络实现手写字识别(low level api)
def conv_layer(input_tensor,name,kernel_size,n_output_channels,padding_mode='SAME',strides=(1,1,1,1)):
    '''卷积层
    :param input_tensor: 输入张量
    :param name: 作用域名称
    :param kernel_size: 核大小
    :param n_output_channels: 输出通道数量(特征谱的数量)
    :param padding_mode: 边界填充模式
    :param strides: 步幅
    :return:
    '''
    with tf.variable_scope(name):
        # 输入张量形状 [batch_size,X width, X height, X channels]
        input_shape = input_tensor.get_shape().as_list()
        # 获取输入通道数
        n_input_channels = input_shape[-1]

        # 获取权重参数形状 [kernel width,kernel height,n_input_channels,n_output_channels]
        weights_shape = list(kernel_size) + [n_input_channels,n_output_channels]

        weights = tf.get_variable(name='_weights',shape=weights_shape)
        print(weights)

        biases = tf.get_variable(name='_biases',initializer=tf.zeros(shape=[n_output_channels]))
        print(biases)

        conv = tf.nn.conv2d(input=input_tensor,filter=weights,
                            strides=strides,padding=padding_mode,
                            name='net_pre-activation')
        print(conv)
        conv = tf.nn.relu(conv,name='activation')
        print(conv)
        return conv

def fc_layer(input_tensor,name,n_output_units,activation_fn=None):
    '''全连接层
    :param input_tensor: 输入张量
    :param name: 作用域名称
    :param n_output_units: 输出单元数量
    :param activation_fn: 激活函数
    :return:
    '''
    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()[1:]
        n_input_units = np.prod(input_shape)# 输入单元的数量
        if len(input_shape) > 1:
            input_tensor = tf.reshape(input_tensor,shape=(-1,n_input_units))

        # 构造权重矩阵
        weights_shape = [n_input_units,n_output_units]
        weights = tf.get_variable(name='_weights',shape=weights_shape)
        print(weights)

        # 偏置单元
        biases = tf.get_variable(name='_biases',initializer=tf.zeros(shape=[n_output_units]))
        print(biases)

        # 全连接层
        layer = tf.matmul(input_tensor,weights)
        print(layer)
        layer = tf.nn.bias_add(layer,biases,name='net_pred-activation')
        print(layer)
        if activation_fn is None:
            return layer
        layer = activation_fn(layer,name='activation')
        print(layer)
        return layer
# 构造CNN
def build_cnn():
    # 创建占位符
    tf_x = tf.placeholder(tf.float32,shape=[None,784],name='tf_x')
    tf_y = tf.placeholder(tf.int32,shape=[None],name='tf_y')

    # 将 x 转换为 4D 张量
    # [batchsize,width,height,1]
    tf_x_image = tf.reshape(tf_x,shape=[-1,28,28,1],name='tf_x_reshaped')

    # 类标 one-hot 化
    tf_y_inehot = tf.one_hot(indices=tf_y,depth=10,dtype=tf.float32,name='tf_y_onehot')

    # 卷积层-1
    print("\nbuilding 1st layer:")
    h1 = conv_layer(tf_x_image,name='conv_1',kernel_size=(5,5),padding_mode='VALID',n_output_channels=32)

    # 子采样-最大化池
    h1_pool = tf.nn.max_pool(h1,ksize=[1,2,2,1],# 各个维度的窗口大小
                             strides=[1,2,2,1],# 步幅
                             padding='SAME')
    # 卷积层-2
    print("\nbuilding 2st layer:")
    h2 = conv_layer(h1_pool,name='conv_2',kernel_size=(5,5),padding_mode='VALID',n_output_channels=64)

    # 子采样-最大化池
    h2_pool = tf.nn.max_pool(h2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    # 全连接层-3
    print("\nbuilding 3st layer:")
    h3 = fc_layer(h2_pool,name='fc_3',n_output_units=100,activation_fn=tf.nn.relu)

    # 丢弃，每次迭代随机以1-keep_prob概率丢弃一些神经元
    keep_prob = tf.placeholder(tf.float32,name='fc_keep_prob')
    h3_drop = tf.nn.dropout(h3,keep_prob=keep_prob,name='dropout_layer')

    # 全连接层（线性激活）-4
    print("\nbuilding 4th layer:")
    h4 = fc_layer(h3_drop,name='fc_4',n_output_units=10,activation_fn=None)

    # 预测
    predictions = {
        'probabilities':tf.nn.softmax(h4,name='probabilities'),
        'labels':tf.cast(tf.argmax(h4,axis=1),tf.int32,name='labels')
    }

    # 可视化图

    # 损失函数和优化器
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h4,labels=tf_y_inehot),
                                        name='cross_entropy_loss')
    # 优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    optimizer = optimizer.minimize(cross_entropy_loss,name='train_op')

    # 计算预测准确率
    correct_predictions = tf.equal(predictions['labels'],tf_y,name='correct_preds')
    accuracy = tf.reduce_mean(tf.cast(correct_predictions,tf.float32),name='accuracy')

def save(saver,sess,epoch,path='./model/'):# 储存模型
    if not os.path.isdir(path):
        os.makedirs(path)
    print("saving model in %s"%path)
    saver.save(sess,os.path.join(path,'cnn-model.ckpt'),global_step=epoch)

def load(saver,sess,path,epoch): # 加载模型
    print("loading model from %s"%path)
    saver.restore(sess,os.path.join(path,'cnn-model.ckpt-%d'%epoch))

# 训练模型
def train(sess,training_set,validation_set=None,initialize=True,
          epochs=20,shuffle=True,dropout=0.5,random_seed=None):
    X_data = np.array(training_set[0])
    y_data = np.array(training_set[1])
    training_loss = []

    if initialize:
        sess.run(tf.global_variables_initializer())
    np.random.seed(random_seed)
    for epoch in range(1,epochs+1):
        batch_gen = batch_generator(X_data,y_data,shuffle=shuffle,batch_size=10)
        avg_loss = 0.0
        for i, (batch_x,batch_y) in enumerate(batch_gen):
            feed = {'tf_x:0':batch_x,'tf_y:0':batch_y,'fc_keep_prob:0':dropout}
            loss, _ = sess.run(['cross_entropy_loss:0','train_op'],feed_dict=feed)
            avg_loss += loss
        training_loss.append(avg_loss/(i+1))
        print("epoch %02d training avg. loss:%7.3f"%(epoch,avg_loss),end=' ')
        if validation_set is not None:
            feed = {'tf_x:0':validation_set[0],'tf_y:0':validation_set[1],'fc_keep_prob:0':1.0}
            valid_acc = sess.run('accuracy:0',feed_dict=feed)
            print(' validation acc: %7.3f'%valid_acc)
        else:
            print()

def predict(sess,X_test,return_proba=False):# 预测
    feed = {'tf_x:0':X_test,'fc_keep_prob:0':1.0}
    if return_proba:
        return sess.run('probabilities:0',feed_dict=feed)
    else:
        return sess.run('labels:0',feed_dict=feed)



def function3():
    X_data, y_data = load_mnist("../../../machinelearndata/chapter12/", 'train')
    print('rows: %d,columns:%d' % (X_data.shape[0], X_data.shape[1]))
    X_test, y_test = load_mnist("../../../machinelearndata/chapter12/", 't10k')
    print('rows: %d,columns:%d' % (X_test.shape[0], X_test.shape[1]))

    X_train,y_train = X_data[:50000,:],y_data[:50000]
    X_valid,y_valid = X_data[50000:,:],y_data[50000:]

    mean_vals = np.mean(X_train,axis=0)
    std_val = np.std(X_train)

    # 聚集样本
    X_train_centered = (X_train-mean_vals)/std_val
    X_valid_centered = (X_valid-mean_vals)/std_val
    X_test_centered = (X_test-mean_vals)/std_val

    # 测试卷积层函数
    g = tf.Graph()
    with g.as_default():
        x = tf.placeholder(tf.float32,shape=[None,28,28,1])
        conv_layer(x,name='convtest',kernel_size=(3,3),n_output_channels=32)
    del g,x

    # 测试全连接层
    g = tf.Graph()
    with g.as_default():
        x = tf.placeholder(tf.float32,shape=[None,28,28,1])
        fc_layer(x,name='fctest',n_output_units=32,activation_fn=tf.nn.relu)
    del g,x

    # 创建图
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(123)
        build_cnn()
        saver = tf.train.Saver()

    # 训练cnn模型
    with tf.Session(graph=g) as sess:
        train(sess,training_set=(X_train_centered,y_train),
              # validation_set=(X_valid_centered,y_valid),
              initialize=True,
              random_seed=123)
        save(saver,sess,epoch=20)

# 恢复储存的模型
def function4():
    X_data, y_data = load_mnist("../../../machinelearndata/chapter12/", 'train')
    print('rows: %d,columns:%d' % (X_data.shape[0], X_data.shape[1]))
    X_test, y_test = load_mnist("../../../machinelearndata/chapter12/", 't10k')
    print('rows: %d,columns:%d' % (X_test.shape[0], X_test.shape[1]))

    X_train, y_train = X_data[:50000, :], y_data[:50000]
    X_valid, y_valid = X_data[50000:, :], y_data[50000:]

    mean_vals = np.mean(X_train, axis=0)
    std_val = np.std(X_train)

    # 聚集样本
    X_train_centered = (X_train - mean_vals) / std_val
    X_valid_centered = (X_valid - mean_vals) / std_val
    X_test_centered = (X_test - mean_vals) / std_val

    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(123)
        # build_cnn()
        # saver = tf.train.Saver()
        saver = tf.train.import_meta_graph('./model/cnn-model.ckpt-20.meta')
    with tf.Session(graph=g) as sess:
        load(saver,sess,epoch=20,path='./model/')
        preds = predict(sess,X_test_centered[:1000],return_proba=False)
        print("test accuracy: %.3f%%"%(100*np.sum(preds==y_test[:1000])/len(y_test[:1000])))




if __name__ == '__main__':
    # function1()
    # function2()
    # function3()
    function4()