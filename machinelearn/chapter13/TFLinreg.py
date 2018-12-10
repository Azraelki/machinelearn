import tensorflow as tf
# 使用tensorflow定义一个成本函数为最小二乘法的模型
class TFLinreg:
    def __init__(self,x_dim,learning_rate=0.01,random_seed=None):
        self.x_dim = x_dim # 特征数
        self.learning_rate = learning_rate
        self.g = tf.Graph()
        # 创建模型
        with self.g.as_default():
            # 设置grapy-level 随机种子
            tf.set_random_seed(random_seed)
            self.build()
            # 创建初始器
            self.init_op = tf.global_variables_initializer()

    def build(self):
        # 为输入定义预留变量
        self.X = tf.placeholder(dtype=tf.float32,
                                shape=(None,self.x_dim),
                                name='x_input')
        self.y = tf.placeholder(dtype=tf.float32,shape=(None),name='y_input')
        print(self.X)
        print(self.y)

        ## 定义参数矩阵和偏差向量
        w = tf.Variable(tf.zeros(shape=(1)),name='weight')
        b = tf.Variable(tf.zeros(shape=(1)),name='bias')
        print(w)
        print(b)

        # 定义 w*x+b 表达式输出
        self.z_net = tf.squeeze(w*self.X+b,name='z_net')
        print(self.z_net)

        # 定义平方差
        sqr_errors = tf.square(self.y-self.z_net,name='sqr_errors')
        print(sqr_errors)

        # 定义均方差(成本函数)
        self.mean_cost = tf.reduce_mean(sqr_errors,name="sqr_cost")

        # 定义梯度下降优化算法
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate,
                                                      name='GradientDescent')
        self.optimizer = optimizer.minimize(self.mean_cost)


