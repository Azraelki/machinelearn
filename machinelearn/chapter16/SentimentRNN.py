import tensorflow as tf

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
            self.bild()
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

        # lstm_outputs 的形状为 [batch_size, max_time, cells.output_size]



