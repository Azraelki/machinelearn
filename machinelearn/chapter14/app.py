import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# 张量
def function1():
    g = tf.Graph()
    # 定义计算图
    with g.as_default():
        # 定义张量
        t1 = tf.constant(np.pi)
        t2 = tf.constant([1,2,3,4])
        t3 = tf.constant([[1,2],[3,4]])

        # 获取张量的等级
        r1 = tf.rank(t1)
        r2 = tf.rank(t2)
        r3 = tf.rank(t3)

        # 获取张量的形状
        s1 = t1.get_shape()
        s2 = t2.get_shape()
        s3 = t3.get_shape()
        print('shapes:',s1,s2,s3)

        with tf.Session(graph=g) as sess:
            print("ranks:",r1.eval(),r2.eval(),r3.eval())

# 理解计算图 z=2*(a-b)+c
def function2():
    g = tf.Graph()
    with g.as_default():# 在此图上下文中定义节点
        a = tf.constant(1,name='a')
        b = tf.constant(2,name='b')
        c = tf.constant(3,name='c')
        z = 2*(a-b)+c
    with tf.Session(graph=g) as sess: # 在此session下运行图
        print("2*(a-b)+c => ",sess.run(z))

# 理解placeholder z=2*(a-b)+c
def function3():
    g = tf.Graph()
    with g.as_default():
        tf_a = tf.placeholder(tf.int32,shape=[],name='tf_a')
        tf_b = tf.placeholder(tf.int32,shape=[],name='tf_b')
        tf_c = tf.placeholder(tf.int32,shape=[],name='tf_c')

        r1 = tf_a - tf_b
        r2 = 2*r1
        z = r2 + tf_c
    with tf.Session(graph=g) as sess:
        feed = {
            tf_a:1,tf_b:2,tf_c:3
        }
        print("z:",sess.run(z,feed_dict=feed))

    # 定义不确定大小的数组
    g = tf.Graph()
    with g.as_default():
        tf_x = tf.placeholder(tf.float32,shape=[None,2],name='tf_x')
        x_mean = tf.reduce_mean(tf_x,axis=0,name='mean')
    np.random.seed(123)
    np.set_printoptions(precision=2)
    with tf.Session(graph=g) as sess:
        x1 = np.random.uniform(low=0,high=1,size=(5,2))
        print("feeding data with shape ",x1.shape)
        print("result:",sess.run(x_mean,feed_dict={tf_x:x1}))

        x2 = np.random.uniform(low=0,high=1,size=(10,2))
        print("feeding data with shape ",x2.shape)
        print("result:",sess.run(x_mean,feed_dict={tf_x:x2}))

# 定义变量
def function4():
    g = tf.Graph()

    with g.as_default():
        w = tf.Variable(np.array([[1,2,3,4],[5,6,7,8]]))
        print(w)
    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer()) # 执行初始化后变量才真正分配内存
        print(sess.run(w)) # 所有的变量定义都应在初始化变量之前


# 变量作用域
def function5():
    g = tf.Graph()
    with g.as_default():
        with tf.variable_scope("net_A"):
            with tf.variable_scope("layer-1"):
                w1 = tf.Variable(tf.random_normal(shape=(10,4)),name="weights")
            with tf.variable_scope("layer-2"):
                w2 = tf.Variable(tf.random_normal(shape=(20,10)),name='weights')
        with tf.variable_scope("net_B"):
            with tf.variable_scope("layer-1"):
                w3 = tf.Variable(tf.random_normal(shape=(10,4)),name="weights")

        print(w1)
        print(w2)
        print(w3)
        # 打印出的变量会显示自己的作用域前缀 <tf.Variable 'net_A/layer-1/weights:0' shape=(10, 4) dtype=float32_ref>

# 重用变量
def build_classifier(data,labels,n_classes=2): # 分类器
    data_shape = data.get_shape().as_list()
    weights = tf.get_variable(name='weights',
                              shape=(data_shape[1],n_classes),
                              dtype=tf.float32)
    bias = tf.get_variable(name='bias',initializer=tf.zeros(shape=n_classes))

    logits = tf.add(tf.matmul(data,weights),bias,name='logits')
    return logits,tf.nn.softmax(logits)

def build_generator(data,n_hidden):# 生成器网络
    data_shape = data.get_shape().as_list()
    w1 = tf.Variable(tf.random_normal(shape=(data_shape[1],n_hidden)),name='w1')

    b1 = tf.Variable(tf.zeros(shape=n_hidden),name='b1')

    hidden = tf.add(tf.matmul(data,w1),b1,name='hidden_pre-activation')
    hidden = tf.nn.relu(hidden,"hidden_pre-activation")

    w2 = tf.Variable(tf.random_normal(shape=(n_hidden,data_shape[1])),name='w2')
    b2 = tf.Variable(tf.zeros(shape=data_shape[1]),name='b2')
    output = tf.add(tf.matmul(hidden,w2),b2,name='output')
    return output,tf.nn.sigmoid(output)

def function6():
    batch_size = 64
    g = tf.Graph()
    with g.as_default():
        tf_X = tf.placeholder(shape=(batch_size,100),dtype=tf.float32,name='tf_X')

        # 创建生成器
        with tf.variable_scope("generator"):
            gen_out1 = build_generator(data=tf_X,n_hidden=50) # 随机生成 （batch_size*100） 矩阵
        # 创建分类器
        with tf.variable_scope("classifier") as scope:
            # 原始数据的分类器
            cls_out1 = build_classifier(data=tf_X,labels=tf.ones(shape=batch_size))

            # 重用分类器的变量,在这次调用build_classifier时，变量跟上次使用的一样，并没有重新创建
            scope.reuse_variables()# 开启在此作用域内变量重用=True，默认为False，为False时，get_variable将每次生成新的变量
            cls_out2 = build_classifier(data=gen_out1[1],labels=tf.zeros(shape=batch_size))

# 创建一个回归模型
def make_random_data():
    # 创建一个数据集来进行训练
    np.random.seed(0)
    x = np.random.uniform(low=-2,high=4,size=200)
    y = []
    for t in x:
        r = np.random.normal(loc=0.0,scale=(0.5+t*t/3),size=None)
        y.append(r)
    return x,1.762*x-0.84+np.array(y)

def function7():
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(123)
        # 占位符
        tf_x = tf.placeholder(shape=(None),dtype=tf.float32,name='tf_x')
        tf_y = tf.placeholder(shape=(None),dtype=tf.float32,name='tf_y')

        # 定义变量，参数w和b
        weight = tf.Variable(tf.random_normal(shape=(1,1),stddev=0.25),name='weight')
        bias = tf.Variable(0.0,name='bias')

        # 创建模型
        y_hat = tf.add(weight*tf_x,bias,name='y_hat')

        # 计算成本
        cost = tf.reduce_mean(tf.square(tf_y-y_hat),name='cost')

        # 训练模型
        optim = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optim.minimize(cost,name='train_op')

        # 储存模型
        saver = tf.train.Saver()


    # 生成随机数据集
    x,y = make_random_data()
    plt.plot(x,y,'o')
    plt.show()
    x_train,y_train = x[:100],y[:100]
    x_test,y_test = x[100:],y[100:]
    # 开启会话训练模型
    n_epochs = 500
    training_costs = []
    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(n_epochs):
            c,_ = sess.run([cost,train_op],feed_dict={tf_x:x_train,tf_y:y_train})
            training_costs.append(c)
            if not e % 50:
                print("epoch %4d: %.4f"%(e,c))
    plt.plot(training_costs)
    plt.show()

    # 使用名字来执行
    n_epochs = 500
    training_costs = []
    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(n_epochs):
            c,_ = sess.run(['cost:0','train_op'],feed_dict={
                'tf_x:0':x_train,
                'tf_y:0':y_train
            })
            training_costs.append(c)
            if not e % 50:
                print("epoch %4d: %.4f" % (e, c))
        # 储存模型
        saver.save(sess,'./trained-model')

    # 恢复模型
    g = tf.Graph()
    x_arr = np.arange(-2, 4, 0.1)
    with tf.Session(graph=g) as sess:
        # 使用.meta文件重建图
        new_saver = tf.train.import_meta_graph("./trained-model.meta")
        # 还原模型
        new_saver.restore(sess,'./trained-model')
        y_pred = sess.run("y_hat:0",feed_dict={
            'tf_x:0':x_arr
        })
    # 可视化预测结果
    plt.figure()
    plt.plot(x_train,y_train,'bo')
    plt.plot(x_test,y_test,'bo',alpha=0.3)
    plt.plot(x_arr,y_pred.T[:,0],'-r',lw=3)
    plt.show()

# 将张量转换为多维数据数组
def function8():
    g = tf.Graph()
    with g.as_default():
        arr = np.array([[1,2,3,3.5],
                        [4,5,6,6.5],
                        [7,8,9,9.5]])
        T1 = tf.constant(arr,name='T1')
        print(T1)
        s = T1.get_shape()
        print("shape of T1 is ",s)
        T2 = tf.Variable(tf.random_normal(shape=s))
        print(T2)
        # get_shape()方法返回的对象不能直接索引操作，需调用as_list()方法先转换为python数组
        T3 = tf.Variable(tf.random_normal(shape=(s.as_list()[0],)))
        print(T3)

        # 张量转换shape
        T4 = tf.reshape(T1,shape=[1,1,-1],name='T4') # -1表示，根据总数量和其他维度进行推断大小
        print(T4)
        T5 = tf.reshape(T1,shape=[1,3,-1],name='T5')
        print(T5)

        # 转置
        T6 = tf.transpose(T5,perm=[2,1,0],name='T6')
        print(T6)
        T7 = tf.transpose(T5,perm=[0,2,1],name='T7')
        print(T7)

        # 分割张量
        t5_splt = tf.split(T5,num_or_size_splits=2,axis=2,name='T8')
        print(t5_splt) # 返回两个[1,3,2]的张量

        # 拼接张量
        t1 = tf.ones(shape=(5,1),dtype=tf.float32,name='t1')
        t2 = tf.zeros(shape=(5,1),dtype=tf.float32,name='t2')
        print(t1)
        print(t2)
        t3 = tf.concat([t1,t2],axis=0,name='t3')
        t4 = tf.concat([t1,t2],axis=1,name='t4')
        print(t3)
        print(t4)

# tensorflow中的控制流
# f = x+y if x < y else x-y
def function9():
    x,y = 1.0,2.0
    g = tf.Graph()
    with g.as_default():
        tf_x = tf.placeholder(dtype=tf.float32,shape=None,name='tf_x')
        tf_y = tf.placeholder(dtype=tf.float32,shape=None,name='tf_y')
        if x < y:# python方式控制流
            res = tf.add(tf_x,tf_y,name='result_add')
        else:
            res = tf.subtract(tf_x,tf_y,name='result_sub')
        print('object:',res)# 从此输出得出graph只包含了add的分支
    with tf.Session(graph=g) as sess:
        print("x < y:%s -> result:%d"%(x<y,res.eval(feed_dict={"tf_x:0":x,"tf_y:0":y})))
        x,y = 2.0,1.0 # 值改变之后，计算的结果仍然是add操作，计算图是静态的，并没有使用控制流
        print("x < y:%s -> result:%d"%(x<y,res.eval(feed_dict={"tf_x:0":x,"tf_y:0":y})))

    x, y = 1.0, 2.0
    g = tf.Graph()
    with g.as_default():
        tf_x = tf.placeholder(dtype=tf.float32, shape=None, name='tf_x')
        tf_y = tf.placeholder(dtype=tf.float32, shape=None, name='tf_y')
        res = tf.cond(tf_x < tf_y,# tensorflow方式的控制流
                      lambda :tf.add(tf_x, tf_y, name='result_add'),
                      lambda :tf.subtract(tf_x, tf_y, name='result_sub'))
        print('object:', res)  # 从此输出得出graph只包含了add的分支
    with tf.Session(graph=g) as sess:
        print("x < y:%s -> result:%d"%(x<y,res.eval(feed_dict={"tf_x:0":x,"tf_y:0":y})))
        x,y = 2.0,1.0 #
        print("x < y:%s -> result:%d"%(x<y,res.eval(feed_dict={"tf_x:0":x,"tf_y:0":y})))

    # tf.case操作，类似于多if else
    f1 = lambda: tf.constant(1)
    f2 = lambda: tf.constant(0)
    result = tf.case([(tf.less(x, y), f1)], default=f2)
    # 循环操作
    i = tf.constant(0)
    threshold = 100
    c = lambda i:tf.less(i,100)
    b = lambda i:tf.add(i,1)
    r = tf.while_loop(cond=c,body=b,loop_vars=[i])

# 通过tensorboard可视化计算图
def function10():
    batch_size = 64
    g = tf.Graph()
    with g.as_default():
        tf_X = tf.placeholder(shape=(batch_size,100),dtype=tf.float32,name='tf_X')
        # 创建生成器
        with tf.variable_scope("generator"):
            gen_out1 = build_generator(tf_X,n_hidden=50)

        # 创建分类器
        with tf.variable_scope("classifier") as scope:
            # 用原始数据创建
            cls_out1 = build_classifier(tf_X,labels=tf.ones(shape=batch_size))

            # 重用变量，并使用生成器数据
            scope.reuse_variables()
            cls_out2 = build_classifier(data=gen_out1[1],labels=tf.zeros(shape=batch_size))

    # 导出计算图
    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        file_writer = tf.summary.FileWriter(logdir='./logs/',graph=g)



if __name__ == '__main__':
    # function1()
    # function2()
    # function3()
    # function4()
    # function5()
    # function6()
    # function7()
    # function8()
    # function9()
    function10()