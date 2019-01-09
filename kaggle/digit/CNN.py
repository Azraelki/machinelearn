import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
'''
kaggle:手写数字识别-CNN实现
'''
class CNN:
    def __init__(self,batch_size=128,epoch=50,lr=0.001,dropout_rate=0.5,shuffle=False,random_state=123):
        self.batch_size = batch_size        # 批大小
        self.epoch = epoch                  # 代数
        self.lr = lr                        # 学习率
        self.dropout_rate = dropout_rate    # 隐藏单元丢弃率
        self.shuffle = shuffle              # 是否重置数据序列

        # 构造图
        g = tf.Graph()
        with g.as_default():
            tf.random.set_random_seed(random_state)
            self.build()
            self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

        self.sess = tf.Session(graph=g)

    def build(self):
        # 创建占位符
        tf_x = tf.placeholder(tf.float32,shape=(None,784),name='tf_x')
        tf_y = tf.placeholder(tf.int32,shape=None,name='tf_y')

        # 创建状态占位符
        is_train = tf.placeholder(tf.bool,shape=None,name='is_train')

        # 将输入tf_x转化为4-D张量
        tf_x_image = tf.reshape(tf_x,shape=(-1,28,28,1),name='tf_x_image')
        # tf_y one-hot 化
        tf_y_onehot = tf.one_hot(indices=tf_y,depth=10,dtype=tf.float32,name='tf_y_onehot')

        # 卷积层-1
        h1 = tf.layers.conv2d(inputs=tf_x_image,
                              filters=32,# 卷积核数量
                              kernel_size=(5,5), # 卷积核的大小
                              activation=tf.nn.relu)
        # 池最大化
        h1_max_out = tf.layers.max_pooling2d(inputs=h1,
                                             pool_size=(2,2),# 取样的池大小
                                             strides=(2,2))# 取样的池步长

        # 卷积层-2
        h2 = tf.layers.conv2d(inputs=h1_max_out,
                              filters=64,
                              kernel_size=(5,5),
                              activation=tf.nn.relu)
        # 池最大化
        h2_max_out = tf.layers.max_pooling2d(inputs=h2,pool_size=(2,2),strides=(2,2))

        ## 全连接层-3
        # 将数据展开
        input_shape = h2_max_out.get_shape().as_list()
        input_units = np.prod(input_shape[1:])
        h2_pool_out = tf.reshape(h2_max_out,shape=(-1,input_units))
        h3 = tf.layers.dense(inputs=h2_pool_out,units=100,activation=tf.nn.relu)

        # 随机丢弃
        h3_drop = tf.layers.dropout(inputs=h3,seed=123,training=is_train)

        # 全连接层-4
        h4 = tf.layers.dense(inputs=h3_drop,units=10,activation=None)

        # 预测
        predictions = {
            'probabilities':tf.nn.softmax(h4,name='probabilities'),
            'labels':tf.cast(tf.argmax(h4,axis=1),tf.int32,name='labels')
        }

        # 损失函数和优化器
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h4,labels=tf_y_onehot),name='cross_entropy_loss')

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        train_op = optimizer.minimize(cross_entropy_loss,name='train_op')

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

    def train(self,train_set,validation_set=None,initialize=True):
        if initialize:
            self.sess.run(self.init_op)

        self.train_cost = []
        X_data = np.array(train_set[0])
        y_data = np.array(train_set[1])
        for epoch in range(1, self.epoch + 1):
            batch_gen = self.batch_generator(X_data, y_data, shuffle=self.shuffle)
            avg_loss = 0.0
            for i,(batch_x,batch_y) in enumerate(batch_gen):
                feed = {'tf_x:0':batch_x,'tf_y:0':batch_y,'is_train:0':True}
                loss, _ =self.sess.run(['cross_entropy_loss:0','train_op'],feed_dict=feed)
                avg_loss += loss
            self.train_cost.append(avg_loss)
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

    def load(self,epoch,path):
        print("loading model from %s" % path)
        self.saver.restore(self.sess,os.path.join(path,'model.ckpt-%d'%epoch))

    def save(self,epoch,path='./model'):
        if not os.path.exists(path):
            os.makedirs(path)
        print('saving model in %s' % path)
        self.saver.save(self.sess,os.path.join(path,'model.ckpt'),global_step=epoch)

    def predict(self, X_test, return_proba=False):
        feed = {'tf_x:0' : X_test,
        'is_train:0' : False} ## for dropout
        if return_proba:
            return self.sess.run('probabilities:0', feed_dict=feed)
        else:
            return self.sess.run('labels:0',feed_dict=feed)


if __name__ == '__main__':
    # 获取数据集
    train_data = pd.read_csv("../../../machinelearndata/kaggle/digit/train.csv", encoding="utf-8")
    test_data = pd.read_csv("../../../machinelearndata/kaggle/digit/test.csv", encoding="utf-8")
    # train_data的数据的第一列为类标
    print(train_data.shape)
    print(test_data.shape)

    # 特征都为0-255的数字类型
    train_y = train_data['label'].values[:np.newaxis]
    train_x = train_data.drop(columns=['label']).values
    test_x = test_data.values

    # 归一化数据
    std = StandardScaler()
    train_x = std.fit_transform(train_x)
    test_x = std.transform(test_x)

    # 划分数据
    train_s_x, test_s_x, train_s_y, test_s_y = train_test_split(train_x, train_y, test_size=0.2, stratify=train_y)

    cnn = CNN(batch_size=64,shuffle=True)
    cnn.train([train_s_x,train_s_y],[test_s_x,test_s_y])
    cnn.save(cnn.epoch)
    # cnn.load(2,path='./model')
    submition = pd.DataFrame(columns=['ImageId','Label'])
    submition['ImageId'] = [i for i in range(1,test_x.shape[0]+1)]
    l = [pd.Series(cnn.predict(test_x[x:x+1000,:])) for x in range(0,test_x.shape[0],1000)]
    l = np.array(l)
    l = l.flatten()
    submition['Label'] = l
    submition.to_csv(path_or_buf='./submition-cnn.csv',index=False)