import tensorflow as tf
import os
import numpy as np
'''
循环神经网络---字符级的语言预测
'''
class CharRnn:
    def __init__(self,num_classes,batch_size=64,num_steps=100,lstm_size=128,
                 num_layers=1,learning_rate=0.001,keep_prob=0.5,grad_clip=5,
                 sampling=False,char2int=None,int2char=None,chars=None):
        '''
        :param num_classes: 独特字符的数量
        :param batch_size: 批大小
        :param num_steps: 步长
        :param lstm_size: lstm单元数
        :param num_layers: 层数
        :param learning_rate: 学习率
        :param keep_prob: 保留率
        :param grad_clip: 梯度截断，防止梯度爆炸
        :param sampling: True采样模型,False训练模式
        :param char2int:字符转int
        :param int2char:int转字符
        :param chars: 唯一字符集
        '''
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.grad_clip = grad_clip
        self.sampling = sampling
        self.char2int= char2int
        self.int2char = int2char
        self.chars = chars

        self.g = tf.Graph()
        with self.g.as_default():
            tf.set_random_seed(123)
            self.build(sampling=self.sampling)
            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()

    def build(self,sampling):
        if sampling:
            batch_size,num_steps = 1,1
        else:
            batch_size = self.batch_size
            num_steps = self.num_steps
        tf_x = tf.placeholder(tf.int32,shape=[batch_size,num_steps],name='tf_x')
        tf_y = tf.placeholder(tf.int32,shape=[batch_size,num_steps],name='tf_y')
        tf_keepprob = tf.placeholder(tf.float32,name='tf_keepprob')

        # 数据转化为one-hot形式
        x_onehot = tf.one_hot(tf_x,depth=self.num_classes)
        y_onehot = tf.one_hot(tf_y,depth=self.num_classes)

        # 创建multi-yaler RNN 单元
        cells = tf.nn.rnn_cell.MultiRNNCell([
            tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size),
                                          output_keep_prob=tf_keepprob)
            for _ in range(self.num_layers)
        ])

        # 定义初始状态
        self.initial_state = cells.zero_state(batch_size,tf.float32)

        # 通过RNN运行每个序列步骤
        lstm_outputs,self.final_state = tf.nn.dynamic_rnn(cells,x_onehot,initial_state=self.initial_state)
        print(" << lstm_output >> ",lstm_outputs)

        seq_output_reshaped = tf.reshape(lstm_outputs,shape=[-1,self.lstm_size],name='seq_output_reshaped')

        logits = tf.layers.dense(inputs=seq_output_reshaped,units=self.num_classes,
                                 activation=None,name='logits')

        proba = tf.nn.softmax(logits,name='probabilities')

        y_reshaped = tf.reshape(y_onehot,shape=[-1,self.num_classes],name='y_reshaped')

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits,
            labels=y_reshaped
        ),name='cost')

        # 梯度截断，防止梯度爆炸
        tvars = tf.trainable_variables()
        grads,_ = tf.clip_by_global_norm(tf.gradients(cost,tvars),self.grad_clip)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.apply_gradients(zip(grads,tvars),name='train_op')

    def create_batch_generator(self,data_x, data_y, num_steps):  # 小批次生成器，每次弹出num_steps长度的数据
        batch_size, tot_batch_length = data_x.shape
        num_batches = int(tot_batch_length / num_steps)
        for b in range(num_batches):
            yield (data_x[:, b * num_steps:(b + 1) * num_steps],
                   data_y[:, b * num_steps:(b + 1) * num_steps])

    def train(self,train_x,train_y,num_epochs,ckpt_dir='./model2/'):
        # 定义模型保存路径
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        with tf.Session(graph=self.g) as sess:
            sess.run(self.init_op)

            n_batches = int(train_x.shape[1]/self.num_steps)
            iterations = n_batches*num_epochs

            for epoch in range(num_epochs):
                # 每代开始时重新初始化state
                new_state = sess.run(self.initial_state)
                loss = 0

                # mini-batch 生成器
                bgen = self.create_batch_generator(train_x,train_y,self.num_steps)
                for b,(batch_x,batch_y) in enumerate(bgen,1):
                    iteration = epoch * n_batches + b
                    feed = {
                        'tf_x:0':batch_x,'tf_y:0':batch_y,
                        'tf_keepprob:0':self.keep_prob,self.initial_state:new_state
                    }
                    batch_cost,_,new_state = sess.run(['cost:0','train_op',self.final_state],
                                                       feed_dict=feed)
                    if iteration % 10 == 0:
                        print("epoch %d/%d iteration %d train loss:%.4f"%(epoch+1,num_epochs,
                                                                          iteration,batch_cost))

                # 保存训练模型
                self.saver.save(sess,os.path.join(ckpt_dir,'language_modeling.ckpt'))

    def sample(self,output_length,ckpt_dir,starter_seq='The '):
        observed_seq = [ch for ch in starter_seq]
        with tf.Session(graph=self.g) as sess:
            self.saver.restore(sess,tf.train.latest_checkpoint(ckpt_dir))
            # 1. 使用starter_seq序列运行模型
            new_state = sess.run(self.initial_state)
            for ch in starter_seq:
                x = np.zeros((1,1))
                x[0,0] = self.char2int[ch]
                feed = {
                    'tf_x:0':x,'tf_keepprob:0':1.0,self.initial_state:new_state
                }
                proba,new_state = sess.run(['probabilities:0',self.final_state],feed_dict=feed)

            ch_id = self.get_top_char(proba,len(self.chars))
            observed_seq.append(self.int2char[ch_id])

            # 2. 使用更新过的observed_seq
            for  i in range(output_length):
                x[0,0] = ch_id
                feed = {'tf_x:0':x,'tf_keepprob:0':1.0,self.initial_state:new_state}
                proba,new_state = sess.run(['probabilities:0',self.final_state],
                                           feed_dict=feed)
                ch_id = self.get_top_char(proba,len(self.chars))
                observed_seq.append(self.int2char[ch_id])

            return ''.join(observed_seq)

    def get_top_char(self,probas,char_size,top_n=5):
        p = np.squeeze(probas)
        p[np.argsort(p)[:-top_n]] = 0.0
        p = p/np.sum(p)
        ch_id = np.random.choice(char_size,1,p=p)[0]
        return ch_id





