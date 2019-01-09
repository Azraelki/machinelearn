import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.contrib import keras
from sklearn.model_selection import train_test_split
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
train_s_x,test_s_x,train_s_y,test_s_y = train_test_split(train_x,train_y,test_size=0.2,stratify=train_y)

# mini-batch生成器
def create_batch_generator(X,y,batch_size=128,shuffle=False):
    X_copy = np.copy(X)
    y_copy = np.copy(y)

    if shuffle:
        data = np.column_stack((X_copy,y_copy))
        np.random.shuffle(data)
        X_copy = data[:,:-1]
        y_copy = data[:,-1]
    for i in range(0,X_copy.shape[0],batch_size):
        yield (X_copy[i:i+batch_size,:],y_copy[i:i+batch_size])

def function1():
    # 1、使用layers接口实现
    # 构造计算图
    g = tf.Graph()
    n_feature = train_x.shape[1]
    n_classes = 10
    with g.as_default():
        tf.random.set_random_seed(123)
        # 构造占位符
        tf_x = tf.placeholder(tf.float32,shape=(None,n_feature),name='tf_x')
        tf_y = tf.placeholder(tf.int32,shape=None,name='tf_y')

        # tf_y one-hot化
        y_onehot = tf.one_hot(indices=tf_y,depth=n_classes)

        # 构造隐藏层
        h1 = tf.layers.dense(tf_x,units=100,activation=tf.tanh,name='layer1')
        h2 = tf.layers.dense(h1,units=100,activation=tf.tanh,name='layer2')

        # 构造输出层
        out = tf.layers.dense(h2,units=10,activation=None,name='layer3')

        # 预测
        prediction = {
            'classes': tf.argmax(out, axis=1, name='predicted_classes'),
            'probabilities': tf.nn.softmax(out, name='softmax_tensor')
        }

        # 定义成本函数
        cost = tf.losses.softmax_cross_entropy(y_onehot,out)
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(cost)
        init_op = tf.global_variables_initializer()

    # 创建会话
    with tf.Session(graph=g) as sess:
        sess.run(init_op) # 初始化参数

        # 迭代50次
        for epoch in range(100):
            train_cost = []
            batch_generator = create_batch_generator(train_s_x,train_s_y,batch_size=64,shuffle=True)
            for batch_x,batch_y in batch_generator:
                # 构造数据参数
                feed_param = {tf_x:batch_x,tf_y:batch_y}
                _,batch_cost = sess.run([train_op,cost],feed_dict=feed_param)
                train_cost.append(batch_cost)
            print('-- epoch %2d avg training loss: %.4f' % (epoch + 1, np.mean(train_cost)))

        # 计算测试误差
        pred = sess.run(prediction['classes'],feed_dict={tf_x:test_s_x})
        print("test accuracy:%.4f"%(np.sum(pred==test_s_y)/test_s_x.shape[0]))

def function2():
    # 2、使用keras实现
    # onehot标签
    y_onehot = keras.utils.to_categorical(train_s_y)

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=50,input_dim=train_s_x.shape[1],
                                 activation='tanh'))
    model.add(keras.layers.Dense(units=50, input_dim=50,
                                 activation='tanh'))
    model.add(keras.layers.Dense(units=y_onehot.shape[1], input_dim=50,
                                 activation='softmax'))

    sgd_optimizer = keras.optimizers.SGD(lr=0.001,momentum=0.9,decay=1e-7)

    # 指定模型的优化器
    model.compile(sgd_optimizer,loss='categorical_crossentropy')

    # 训练数据
    history = model.fit(train_s_x,y_onehot,batch_size=64,validation_split=0.1,epochs=50)

    # 预测
    pred = model.predict_classes(test_s_x)
    print("test accuracy:%.4f"%(np.sum(pred==test_s_y)/test_s_x.shape[0]))

    # 生成submition
    submition = pd.DataFrame(columns=['ImageId','Label'])
    submition['ImageId'] = [i for i in range(1,test_x.shape[0]+1)]
    submition['Label'] = model.predict_classes(test_x)
    submition.to_csv("./submition.csv",index=False)


function2()


