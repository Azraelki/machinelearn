import tensorflow as tf
import numpy as np
'''
循环神经网络-情感分析
'''
class SentimentRNN:
    def __init__(self,n_words,seq_len=200,lstm_size=256,num_layers=1,batch_size=64,
                 learning_rate=0.0001,embed_size=200):
        '''
        :param n_words: 唯一单词数量+1，（0标识填充值），在构建嵌入层时与embed_size配合使用
        :param seq_len: 每条序列的固定长度（预处理时设定的长度）
        :param lstm_size: 每个RNN层中隐藏单元的数量
        :param num_layers: RNN层数
        :param batch_size: 批大小
        :param learning_rate: 学习速率
        :param embed_size: 嵌入大小
        '''
        self.n_words = n_words
        self.seq_len = seq_len
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.embed_size = embed_size

        self.g = tf.Graph()
        with self.g.as_default():
            tf.set_random_seed(123)
            self.build()
            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()

    def build(self):
        # 定义占位符
        tf_x = tf.placeholder(tf.int32,shape=(self.batch_size,self.seq_len),name='tf_x')
        tf_y = tf.placeholder(tf.float32,shape=(self.batch_size),name='tf_y')
        tf_keepprob = tf.placeholder(tf.float32,name='tf_keepprob')# 保存率

        # 创建嵌入层
        embedding = tf.Variable(tf.random_uniform((self.n_words,self.embed_size),minval=-1,maxval=1))
        embed_x = tf.nn.embedding_lookup(embedding,tf_x,name='embeded_x')

        # 定义 LSTM 单元并堆积到一块
        cells = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size),
                                           output_keep_prob=tf_keepprob)
             for i in range(self.num_layers)
             ]
        )

        # 定义初始状态
        self.initial_state = cells.zero_state(self.batch_size,tf.float32)
        print("<< initial state >> ",self.initial_state)

        lstm_outputs,self.final_state = tf.nn.dynamic_rnn(cells,embed_x,
                                                          initial_state=self.initial_state)

        # lstm_outputs 的形状为 [batch_size, num_steps, lstm_size]
        print("\n <<lstm_output  >>",lstm_outputs)
        print("\n << final state  >>",self.final_state)

        logits = tf.layers.dense(inputs=lstm_outputs[:,-1],units=1,
                                 activation=None,name='logits')
        logits = tf.squeeze(logits,name='logits_squeezed')
        print("\n  << logits   >>",logits)

        y_proba = tf.nn.sigmoid(logits,name='probabolities')
        predictions = {
            'probabilities':y_proba,
            'labels':tf.cast(tf.round(y_proba),tf.int32,name='labels')
        }
        print("\n << predictions >>",predictions)

        # 定义成本函数
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_y,
                                                                      logits=logits),name='cost')

        # 定义优化器
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(cost,name='train_op')


    def create_batch_generator(self,x, y=None, batch_size=64):
        # 定义一个生成器--生成mini-barches
        n_batches = len(x) // batch_size
        x = x[:n_batches * batch_size]
        if y is not None:
            y = y[:n_batches * batch_size]
        for ii in range(0, len(x), batch_size):
            if y is not None:
                yield x[ii:ii + batch_size], y[ii:ii + batch_size]
            else:
                yield x[ii:ii + batch_size]

    def train(self,X_train,y_train,num_epochs):
        with tf.Session(graph=self.g) as sess:
            sess.run(self.init_op)
            iteration = 1
            for epoch in range(num_epochs):
                state = sess.run(self.initial_state) # 每代开始时重新更新state
                for batch_x,batch_y in self.create_batch_generator(X_train,y_train,self.batch_size):
                    feed = {
                        'tf_x:0':batch_x,'tf_y:0':batch_y,
                        'tf_keepprob:0':0.5,self.initial_state:state # 下一批次使用上一批次的state
                    }
                    loss, _, state = sess.run(['cost:0','train_op',self.final_state],feed_dict=feed)

                    if iteration % 20 == 0:
                        print("epoch: %d/%d iteration: %d train loss: %.5f"%(epoch+1,num_epochs,iteration,loss))
                    iteration += 1
                if (epoch+1)%10 == 0:
                    self.saver.save(sess,'model/sentiment-%d.ckpt'%epoch)

    def predict(self,X_data,return_proba=False):
        preds = []
        with tf.Session(graph=self.g) as sess:
            self.saver.restore(sess,tf.train.latest_checkpoint('./model/'))
        test_state = sess.run(self.initial_state)
        for ii, batch_x in enumerate(self.create_batch_generator(X_data,None,batch_size=self.batch_size),1):
            feed = {'tf_x:0':batch_x,'tf_keepprob:0':1.0,self.initial_state:test_state}
            if return_proba:
                pred,test_state = sess.run(['probabilities:0',self.initial_state],feed_dict=feed)
            else:
                pred,test_state = sess.run(['labels:0',self.final_state],feed_dict=feed)
            preds.append(pred)
        return np.concatenate(preds)



