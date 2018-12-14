import tensorflow as tf
import numpy as np
import os
import struct
'''
使用tensorflow的layers层api实现手写字识别卷积神经网络
'''
class ConvNN:
    def __init__(self,batchsize=64,epochs=20,learning_rate=1e-4,dropout_rate=0.5,shuffle=True,random_seed=None):
        '''
        :param batchsize: 批大小
        :param epochs: 迭代次数
        :param learning_rate: 学习速率
        :param dropout_rate: 丢弃率
        :param shuffle: 是否重置数据集
        :param random_seed: 随机种子
        '''
        np.random.seed(random_seed)
        self.batchsize = batchsize
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.shuffle = shuffle

        # 创建图
        g = tf.Graph()
        with g.as_default():
            tf.set_random_seed(random_seed)
            # 构造神经网络
            self.build()
            # 初始化
            self.init_op = tf.global_variables_initializer()
            # 储存器
            self.saver = tf.train.Saver()
        # 创建会话
        self.sess = tf.Session(graph=g)

    def build(self):
        tf_x = tf.placeholder(tf.float32,shape=[None,784],name='tf_x')
        tf_y = tf.placeholder(tf.int32,shape=[None],name='tf_y')
        is_train = tf.placeholder(tf.bool,shape=(),name='is_train')

        # 将x转为4D的张量
        # [batchsize,width,height,1]
        tf_x_image = tf.reshape(tf_x,shape=[-1,28,28,1],name='input_x_2dimages')
        # 类标转为onehot形式
        tf_y_onehot = tf.one_hot(indices=tf_y,depth=10,dtype=tf.float32,name='input_y_onehot')

        # 卷积层-1
        h1 = tf.layers.conv2d(tf_x_image,kernel_size=(5,5),filters=32,activation=tf.nn.relu)
        # 子采样（池最大化）
        h1_pool = tf.layers.max_pooling2d(h1,pool_size=(2,2),strides=(2,2))

        # 卷积层-2
        h2 = tf.layers.conv2d(h1_pool,kernel_size=(5,5),filters=64,activation=tf.nn.relu)
        # 子采样（池最大化）
        h2_pool = tf.layers.max_pooling2d(h2,pool_size=(2,2),strides=(2,2))

        # 全连接层-3
        input_shape = h2_pool.get_shape().as_list()
        n_input_units = np.prod(input_shape[1:])
        h2_pool_flat = tf.reshape(h2_pool,shape=[-1,n_input_units])
        h3 = tf.layers.dense(h2_pool_flat,units=100,activation=tf.nn.relu)

        # 每次迭代按照丢弃率丢弃隐藏单元(只在is_train为true时生效)
        h3_drop = tf.layers.dropout(h3,rate=self.dropout_rate,training=is_train)

        # 全连接-4（线性激活）
        h4 = tf.layers.dense(h3_drop,units=10,activation=None)

        # 预测
        predictions = {
            'probabilities':tf.nn.softmax(h4,name='probabilities'),
            'labels':tf.cast(tf.argmax(h4,axis=1),tf.int32,name='labels')
        }

        # 损失函数和优化器
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h4,labels=tf_y_onehot),
                                            name='cross_entropy_loss')

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = optimizer.minimize(cross_entropy_loss,name='train_op')

        ## 寻找精确度
        correct_predictions = tf.equal(
            predictions['labels'],
            tf_y, name='correct_preds')
        accuracy = tf.reduce_mean(
            tf.cast(correct_predictions, tf.float32),
            name='accuracy')

    def batch_generator(self,X, y, batch_size=64, shuffle=False, random_seed=None):
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
        for i in range(0, X.shape[0], batch_size):
            yield (X[i:i + batch_size, :], y[i:i + batch_size])

    def save(self,epoch,path='./tflayers-model/'):
        if not os.path.isdir(path):
            os.makedirs(path)
        print('saving model in %s'%path)
        self.saver.save(self.sess,os.path.join(path,'model.ckpt'),global_step=epoch)

    def load(self,epoch,path):
        print("loading model from %s"% path)
        self.saver.restore(self.sess,os.path.join(path,'model.ckpt-%d'%epoch))

    def train(self,training_set,validation_set=None,initialize=True):
        if initialize:
            self.sess.run(self.init_op)
        self.train_cost_ = []
        X_data = np.array(training_set[0])
        y_data = np.array(training_set[1])

        for epoch in range(1,self.epochs+1):
            batch_gen = self.batch_generator(X_data,y_data,shuffle=self.shuffle)
            avg_loss = 0.0
            for i,(batch_x,batch_y) in enumerate(batch_gen):
                feed = {'tf_x:0':batch_x,'tf_y:0':batch_y,'is_train:0':True}
                loss, _ =self.sess.run(['cross_entropy_loss:0','train_op'],feed_dict=feed)
                avg_loss += loss
            print("epoch %02d: trraining avg. loss: %7.3f"%(epoch,avg_loss),end=' ')

            if validation_set is not None:
                feed = {'tf_x:0': batch_x,
                        'tf_y:0': batch_y,
                        'is_train:0': False}  ## for dropout
                valid_acc = self.sess.run('accuracy:0',
                                          feed_dict=feed)
                print('Validation Acc: %7.3f' % valid_acc)
            else:
                print()

    def predict(self, X_test, return_proba=False):
        feed = {'tf_x:0' : X_test,
        'is_train:0' : False} ## for dropout
        if return_proba:
            return self.sess.run('probabilities:0', feed_dict=feed)
        else:
            return self.sess.run('labels:0',feed_dict=feed)

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

if __name__ == '__main__':
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

    # cnn = ConvNN(random_seed=123)
    # cnn.train(training_set=(X_train_centered,y_train),initialize=True)
    # cnn.save(epoch=20)
    #
    # del cnn
    cnn2 = ConvNN(random_seed=123)
    cnn2.load(epoch=20,path='./tflayers-model/')
    print(cnn2.predict(X_test_centered[:10,:]))

    preds = cnn2.predict(X_test_centered[:1000,:])
    print("test accuracy:%.2f%%"%(100*np.sum(y_test[:1000]==preds)/len(y_test[:1000])))

