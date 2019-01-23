'''
ImageNet----基于tensorflow的AlexNet构建
论文参考：http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
网络架构：5层卷积层+3层全连接层
架构描述：
    1、第2,4,5层的过滤器只与同一GPU上的前一层特征池进行连接
    2、第3层卷积层与所有的特征池连接
    3、全连接层与前一层所有神经元连接
    4、响应归一化层跟随在第1,2卷积层之后
    5、最大化池层跟在响应归一化层和第五层之后
    6、ReLU跟在所有的卷积层和全连接层
    7、在前全连接的前两层使用drop
权重描述:
    层             过滤器（数量，步幅）        全连接层(神经元数量)           响应归一化      最大化池（步幅）
    1               11*11*3（48*2，4）                                             是               3*3（2）
    2               5*5*48（128*2,1）                                              是               3*3（2）
    3               3*3*256（192*2,1）                                             否                无
    4               3*3*192（192*2,1）                                             否                无
    5               3*3*192（256,1）                                               否               3*3（2）
    6                                           2048*2                             否                无
    7                                           2048*2                             否                无
    8                                           1000                               否                无

'''
import numpy as np
import tensorflow as tf
import os

class AlexNet:
    def __init__(self,batch_size=128,epoch=50,lr=0.01,drop_rate=0.5,n_input=224*224*3,input_channel=3,n_label=1000,shuffle=True,random_state=123):
        self.batch_size = batch_size      # 批大小
        self.epoch = epoch                # 迭代次数
        self.lr = lr                      # 学习率
        self.drop_rate = drop_rate        # drop_out层丢弃率
        self.n_iput = n_input             # 单个样本的维数
        self.input_channel = input_channel# 单个样本通道数
        self.n_label = n_label            # 标签的维数
        self.shuffle = shuffle            # 每次迭代时是否重置数据排序
        self.random_state = random_state  # 随机种子
        # 构造计算图
        self.g_ = tf.Graph()
        with self.g_.as_default():
            tf.random.set_random_seed(self.random_state)
            self.build()
            self.init_op_ = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        self.sess_ = tf.Session(graph=self.g_)

    def build(self):
        tf_x = tf.placeholder(tf.float32,shape=(None,self.n_iput),name='tf_x')
        tf_y = tf.placeholder(tf.int32,shape=None,name='tf_y')
        # 创建状态占位符
        is_train = tf.placeholder(tf.bool, shape=None, name='is_train')

        # 将tf_x转化为4D张量
        width = int(np.sqrt(self.n_iput/self.input_channel))
        tf_x_4d = tf.reshape(tf_x,shape=(-1,width,width,self.input_channel),name='tf_x_4d')
        # 将标签onehot化
        tf_y_onehot = tf.one_hot(indices=tf_y,depth=self.n_label,dtype=tf.float32,name='tf_y_onehot')

        ## 卷积层-1
        h1 = tf.layers.conv2d(tf_x_4d,filters=96,kernel_size=(11,11),
                              strides=(4,4),activation=tf.nn.relu,name='h1')
        norm1 = tf.nn.local_response_normalization(h1,depth_radius=5,bias=2,alpha=1e-4,beta=0.75,name='norm1')
        max_out1 = tf.layers.max_pooling2d(norm1,pool_size=(3,3),strides=(2,2),name='max_out1')

        ## 卷积层-2
        h2 = tf.layers.conv2d(max_out1,filters=256,kernel_size=(5,5),
                              strides=(1,1),activation=tf.nn.relu,name='h2')
        norm2 = tf.nn.local_response_normalization(h2, depth_radius=5, bias=2, alpha=1e-4, beta=0.75, name='norm2')
        max_out2 = tf.layers.max_pooling2d(norm2, pool_size=(3, 3), strides=(2, 2), name='max_out2')

        ## 卷积层-3
        h3 = tf.layers.conv2d(max_out2,filters=384,kernel_size=(3,3),
                              strides=(1,1),activation=tf.nn.relu,name='h3')
        ## 卷积层-4
        h4 = tf.layers.conv2d(h3,filters=384,kernel_size=(3,3),
                              strides=(1,1),activation=tf.nn.relu,name='h4')
        ## 卷积层-5
        h5 = tf.layers.conv2d(h4, filters=256, kernel_size=(3, 3),
                              strides=(1, 1), activation=tf.nn.relu, name='h5')
        max_out5 = tf.layers.max_pooling2d(h5,pool_size=(3,3),strides=(2,2),name='max_ot5')

        # 展开
        input_shape = max_out5.get_shape().as_list()
        units = np.prod(input_shape[1:])
        max_pool_out5 = tf.reshape(max_out5,shape=(-1,units),name='max_pool_out5')

        # 全连接层-6
        h6 = tf.layers.dense(max_pool_out5,units=4096,activation=tf.nn.relu,name='h6')
        drop_out6 = tf.layers.dropout(h6,rate=self.drop_rate,training=is_train,name='drop_out6')
        # 全连接层-7
        h7 = tf.layers.dense(drop_out6,units=4096,activation=tf.nn.relu,name='h7')
        drop_out7 = tf.layers.dropout(h7,rate=self.drop_rate,training=is_train,name='drop_out7')
        # 全连接层-8
        h8 = tf.layers.dense(drop_out7,units=self.n_label,activation=tf.nn.relu,name='h8')

        # 预测
        predictions = {
            'probabilities':tf.nn.softmax(h8,name='probabilities'),
            'labels':tf.cast(tf.argmax(h8,axis=1),tf.int32,name='labels')
        }
        # 精准度
        correct_prediction = tf.equal(predictions['labels'],tf_y,
                                      name='correct_prediction')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),
                                  name='accuracy')

        # 成本函数
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_y,logits=h8),name='cost')
        # 优化器
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        train_op = optimizer.minimize(cost,name='train_op')

    def train(self,train_x,train_y,initialize=True):
        if initialize:
            self.sess_.run(self.init_op_)
        self.train_costs_ = []
        for epoch in range(self.epoch):
            avg_cost = 0.0
            batch_generator = self.create_batch_generator(train_x,train_y)
            for batch_x,batch_y in batch_generator:
                feed_param = {
                    'tf_x:0':batch_x,
                    'tf_y:0':batch_y,
                    'is_train:0':True
                }
                c,_ = self.sess_.run(['cost:0','train_op'],feed_param=feed_param)
                avg_cost += c
            self.train_costs_.append(avg_cost/self.batch_size)
            print("epoch-{} train_cost:{}".format(epoch,self.train_costs_[-1]))



    def export_graph(self):# 导出tensorboard
        self.sess_.run(self.init_op_)
        file_writer = tf.summary.FileWriter(logdir='./logs/', graph=self.g_)

    def create_batch_generator(self,train_x,train_y):
        # 数据生成器，从磁盘中读取数据并处理
        X = np.copy(train_x)
        y = np.copy(train_y)
        if self.shuffle:
            indices = np.random.shuffle(np.arange(0,train_x.shape[0]))
            X = X[indices]
            y = y[indices]
        for i in range(0,train_x.shape[0],self.epoch):
            yield (X[i:i+self.epoch,::],y[i:i+self.epoch])

    def save(self,epoch,path='./model'):# 保存模型
        if not os.path.exists(path):
            os.mkdir(path)
        print('saving model in %s'%path)
        self.saver.save(self.sess_,os.path.join(path,'model.ckpt'),global_step=epoch)

    def load(self,epoch,path='./model'):# 加载模型
        print('loading model from %s'%path)
        self.saver.restore(self.sess_,save_path=os.path.join(path,'model.ckpt-%d'%(epoch)))

    def prediction(self,x_test,return_probabilities=False):# 预测
        feed_param = {
            'tf_x:0':x_test,
            'is_train':False
        }
        if return_probabilities:
            return self.sess_.run(['probabilities:0'],feed_dict=feed_param)
        else:
            return self.sess_.run(['labels:0'],feed_dict=feed_param)


if __name__ == '__main__':
    alex_net = AlexNet()
    # alex_net.export_graph()


