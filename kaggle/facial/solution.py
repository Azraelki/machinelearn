'''
kaggle: facial keypoints detection

'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as  plt


# 加载数据

def load_data():
    train = pd.read_csv('../../../machinelearndata/kaggle/facial/training.csv')
    test = pd.read_csv('../../../machinelearndata/kaggle/facial/test.csv')

    # 把图片的数据信息转换为数组
    train['Image'] = train['Image'].apply(lambda im:np.fromstring(im,sep=' '))
    test['Image'] = test['Image'].apply(lambda im:np.fromstring(im,sep=' '))

    # 打印每列的值数量
    print('train data count:\n',train.count())
    print("test data count:\n",test.count())

    # 丢弃缺失数据的行
    train = train.dropna()
    test = test.dropna()

    # 获取数据，并将图片数据RGB值转换到（0,1）上
    X_train = np.vstack(train['Image'].values)/255
    X_train = X_train.astype(np.float32)
    X_test = np.vstack(test['Image'].values)/255
    X_test = X_test.astype(np.float32)

    # 获取目标值
    y = train.iloc[:,:-1].values
    y = (y-48)/48 # 将值转换到（-1,1）上
    y = y.astype(np.float32)

    X_train,y = shuffle(X_train,y,random_state=123)
    # 返回读取的数据(训练集，训练目标，测试集)
    return X_train,y,X_test

X_train,y_train,X_test = load_data()
print("X_train shape=={};y_train shape=={}".format(X_train.shape,y_train.shape))
print("X_test shape=={}".format(X_test.shape))

def batch_generator(X,y,batch_size=64):
    X_copy = np.copy(X)
    y_copy = np.copy(y)

    for i in range(0,X.shape[0],batch_size):
        yield (X_copy[i:batch_size,:],y_copy[i:batch_size,:])

# 训练一个简单的单层网络
def single_layer_net():
    # 构造计算图
    g = tf.Graph()
    with g.as_default():
        # 输入占位符
        tf_x = tf.placeholder(tf.float32,shape=(None,9216),name='tf_x')
        tf_y = tf.placeholder(tf.float32,shape=(None,30),name='tf_y')
        # 隐藏层-1
        h1 = tf.layers.dense(inputs=tf_x,units=100,activation=tf.nn.relu,name='h1')
        # 输出层
        h2 = tf.layers.dense(inputs=h1,units=30,activation=None,name='h2')

        # 定义成本函数
        cost = tf.losses.mean_squared_error(labels=tf_y,predictions=h2)
        # 预测
        prediction = {
            'keypoint':tf.add(tf.multiply(h2,tf.constant(48,dtype=tf.float32)),tf.constant(48,dtype=tf.float32),name='prediction')
        }
        # 定义优化器
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01,beta1=0.9,beta2=0.999)
        train_op = optimizer.minimize(loss=cost)
        init_op = tf.global_variables_initializer()

    # 分割数据集
    X_s_train,X_s_test,y_s_train,y_s_test = train_test_split(X_train,y_train,test_size=0.2,random_state=123)
    # 创建会话
    with tf.Session(graph=g) as sess:
        tf.random.set_random_seed(123)
        sess.run(init_op)
        costs = []
        valid_costs = []
        # 迭代
        for epoch in range(400):
            generator = batch_generator(X_s_train,y_s_train)
            avg_cost = 0
            for batch_x,batch_y in generator:
                feed = {
                    tf_x:batch_x,
                    tf_y:batch_y
                }
                _,c = sess.run([train_op,cost],feed_dict=feed)
                avg_cost += c
            costs.append(avg_cost)
            v_c = sess.run([cost],feed_dict={tf_x:X_s_test,tf_y:y_s_test})
            valid_costs.append(v_c)
            print('epoch-{} train_cost:{} valid_cosst:{}'.format(epoch+1,avg_cost,v_c))
        saver = tf.train.Saver()
        saver.save(sess,'./model/model.ckpt',global_step=400)
        plt.plot([i for i in range(1,401)],costs,label='train_loss')
        plt.plot([i for i in range(1,401)],valid_costs,label='valid_coss')
        plt.xlabel("epoch")
        plt.ylabel('MSE')
        plt.ylim(1e-3,1e2)
        plt.yscale("log")
        plt.show()
# 测试效果
def show_on_image(path='./model',epoch=400):
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        tf.train.import_meta_graph(os.path.join(path,'model.ckpt-%d.meta'%(epoch)))
        saver = tf.train.Saver()
        saver.restore(sess,os.path.join(path,'model.ckpt-%d'%(epoch)))
        y_pred = sess.run(['prediction:0'],feed_dict={'tf_x:0':X_test})
        y_pred = y_pred[0]
    fig = plt.figure(figsize=(6,6))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
    for i in range(16):
        ax = fig.add_subplot(4,4,i+1,xticks=[],yticks=[])
        img = X_test[i].reshape(96,96)
        ax.imshow(img,cmap='gray')
        print(y_pred[i][0::2])
        print(y_pred[i][0::2])
        ax.scatter(y_pred[i][0::2],y_pred[i][1::2],marker='x',s=10)
    plt.show()
    l = np.array(y_pred)
    l = l.flatten()
    l = np.reshape(l, (1783, 30))
    l = pd.DataFrame(l, columns=['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y',
                                 'left_eye_inner_corner_x', 'left_eye_inner_corner_y', 'left_eye_outer_corner_x',
                                 'left_eye_outer_corner_y', 'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
                                 'right_eye_outer_corner_x', 'right_eye_outer_corner_y', 'left_eyebrow_inner_end_x',
                                 'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
                                 'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y', 'right_eyebrow_outer_end_x',
                                 'right_eyebrow_outer_end_y', 'nose_tip_x', 'nose_tip_y', 'mouth_left_corner_x',
                                 'mouth_left_corner_y', 'mouth_right_corner_x', 'mouth_right_corner_y',
                                 'mouth_center_top_lip_x', 'mouth_center_top_lip_y', 'mouth_center_bottom_lip_x',
                                 'mouth_center_bottom_lip_y']
                     )

    # 提交
    idlook = pd.read_csv('../../../machinelearndata/kaggle/facial/IdLookupTable.csv')
    locations = []
    for i in range(idlook.shape[0]):
        imageId = idlook.iloc[i]['ImageId']
        feature_name = idlook.iloc[i]['FeatureName']
        location = l.iloc[imageId - 1][feature_name]
        locations.append(location)
        print(imageId)
    idlook['Location'] = locations
    idlook.to_csv('./submition.csv', columns=['RowId', 'Location'], index=False)


# single_layer_net()
# show_on_image()
# single_layer_net()







