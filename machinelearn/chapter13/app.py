import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import struct
import tensorflow.contrib.keras as keras
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
def create_batch_generator(X,y,batch_size=128,shuffle=False):
    ''' 批数据生成器
    :param X:
    :param y:
    :param batch_size: 批大小
    :param shuffle: 是否重置排序
    :return:
    '''
    X_copy = np.copy(X)
    y_copy = np.copy(y)

    if shuffle:
        data = np.column_stack((X_copy,y_copy))
        np.random.shuffle(data)
        X_copy = data[:,:-1]
        y_copy = data[:,-1].astype(int)
    for i in range(0,X.shape[0],batch_size):
        yield (X_copy[i:i+batch_size,:],y_copy[i:i+batch_size])
def function4():
    X_train, y_train = load_mnist("../../../machinelearndata/chapter12/", 'train')
    print('rows: %d,columns:%d' % (X_train.shape[0], X_train.shape[1]))
    X_test, y_test = load_mnist("../../../machinelearndata/chapter12/", 't10k')
    print('rows: %d,columns:%d' % (X_test.shape[0], X_test.shape[1]))

    mean_vals = np.mean(X_train,axis=0)
    std_val = np.std(X_train)

    # 聚集样本的特征值
    X_train_centered = (X_train-mean_vals)/std_val
    X_test_centered = (X_test-mean_vals)/std_val

    del X_train, X_test

    n_features = X_train_centered.shape[1]
    n_classes = 10
    random_seed = 123
    np.random.seed(random_seed)

    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(random_seed)
        # 构造占位符
        tf_x = tf.placeholder(dtype=tf.float32,shape=(None,n_features),name='tf_x')
        tf_y = tf.placeholder(dtype=tf.int32,shape=None,name='tf_y')
        # 构造onehot形式类标
        y_onehot = tf.one_hot(indices=tf_y,depth=n_classes)

        # 构建隐藏层,激活函数使用双曲正切函数
        h1 = tf.layers.dense(inputs=tf_x,units=50,activation=tf.tanh,name='layer1')
        h2 = tf.layers.dense(inputs=h1,units=50,activation=tf.tanh,name='layer2')

        # 构建输出层
        logits = tf.layers.dense(inputs=h2,units=10,activation=None,name='layer3')


        predictions = {
            'classes':tf.argmax(logits,axis=1,name='predicted_classes'),
            'probabilities':tf.nn.softmax(logits,name='softmax_tensor')
        }

        # 定义成本函数和优化器
        cost = tf.losses.softmax_cross_entropy(onehot_labels=y_onehot,logits=logits)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

        train_op = optimizer.minimize(loss=cost)

        init_op = tf.global_variables_initializer()

    # 创建一个新的会话来启动graph
    sess = tf.Session(graph=g)
    # 执行变量初始化操作
    sess.run(init_op)

    # 迭代50次训练模型
    for epoch in range(50):
        train_costs = []
        batch_generatot = create_batch_generator(X_train_centered,y_train,batch_size=64,shuffle=True)
        for batch_X,batch_y in batch_generatot:
            # 构造字典向网络填充数据
            feed = {tf_x:batch_X,tf_y:batch_y}
            _,batch_cost = sess.run([train_op,cost],feed_dict=feed)
            train_costs.append(batch_cost)
        print('-- epoch %2d avg training loss: %.4f'%(epoch+1,np.mean(train_costs)))

    # 在测试集上预测
    feed = {tf_x:X_test_centered}
    y_pred = sess.run(predictions['classes'],feed_dict=feed)
    print("test accuracy:%.2f%%"%(100*np.sum(y_pred==y_test)/y_test.shape[0]))

# 使用keras实现手写字识别
def function5():
    X_train, y_train = load_mnist("../../../machinelearndata/chapter12/", 'train')
    print('rows: %d,columns:%d' % (X_train.shape[0], X_train.shape[1]))
    X_test, y_test = load_mnist("../../../machinelearndata/chapter12/", 't10k')
    print('rows: %d,columns:%d' % (X_test.shape[0], X_test.shape[1]))

    mean_vals = np.mean(X_train, axis=0)
    std_val = np.std(X_train)

    # 聚集样本的特征值
    X_train_centered = (X_train - mean_vals) / std_val
    X_test_centered = (X_test - mean_vals) / std_val

    del X_train, X_test

    # 将类标转换为onehot形式
    y_train_onehot = keras.utils.to_categorical(y_train)
    print("y_train_onehot shape",y_train_onehot.shape)

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=50,
                                 input_dim=X_train_centered.shape[1],
                                 kernel_initializer='glorot_uniform',
                                 bias_initializer='zeros',
                                 activation='tanh'))
    model.add(keras.layers.Dense(units=50,
                                 input_dim=50,
                                 kernel_initializer='glorot_uniform',# 泽维尔初始化
                                 bias_initializer='zeros',
                                 activation='tanh'))
    model.add(keras.layers.Dense(units=y_train_onehot.shape[1],
                                 input_dim=50,
                                 kernel_initializer='glorot_uniform',
                                 bias_initializer='zeros',
                                 activation='softmax'))
    sgd_optimizer = keras.optimizers.SGD(lr=0.001,
                                         decay=1e-7,# 学习速率衰减系数
                                         momentum=0.9)
    # 编译，指定优化器函数为随机梯度下降，损失函数为分类交叉熵
    model.compile(optimizer=sgd_optimizer,loss='categorical_crossentropy') # 分类交叉熵，是多类别预测对softmax的推广

    # 训练数据
    history = model.fit(X_train_centered,y_train_onehot,
                        batch_size=64, # 批的大小
                        epochs=50,# 迭代次数
                        verbose=1, # 打印信息的详细程度
                        validation_split=0.1) # 每次迭代保留10%样本用于校验是否在训练中过拟合

    # 预测
    y_train_pred = model.predict_classes(X_train_centered,verbose=0)
    print("first 3 predictions:",y_train_pred[:3])

    # 在训练集和测试集上计算精准度
    y_train_pred = model.predict_classes(X_train_centered,verbose=0)
    correct_preds = np.sum(y_train==y_train_pred,axis=0)
    train_acc = correct_preds/y_train.shape[0]
    print("training accuracy:%.2f%%"%(train_acc*100))

    y_test_pred = model.predict_classes(X_test_centered,verbose=0)
    correct_preds = np.sum(y_test==y_test_pred,axis=0)
    test_acc = correct_preds/y_test.shape[0]
    print("test accuracy:%.2f%%"%(test_acc*100))









if __name__ == '__main__':
    # function1()
    # function2()
    # function3()
    # function4()
    function5()