import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as  plt
from sklearn.model_selection import train_test_split
import os
from solution import X_train, y_train, X_test
'''
kaggle-使用CNN实现面部关键点检测(facial-keypoints)
'''
class CNN:
    def __init__(self,batch_size=128,epoch=400,lr=0.01,drop_rate=0.5,shuffle=False,random_state=123):
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr
        self.drop_rete = drop_rate
        self.shuffle = shuffle
        g = tf.Graph()
        with g.as_default():
            tf.random.set_random_seed(random_state)
            self.build()
            self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=g)

    def build(self):
        # 构造计算图
        tf_x = tf.placeholder(tf.float32,(None,9216),'tf_x')
        tf_y = tf.placeholder(tf.float32,(None,30),'tf_y')
        # 使用tf_x构造4D张量
        tf_x_4d = tf.reshape(tf_x,shape=(-1,96,96,1),name='tf_x_4d')
        # 创建占位符is_train
        is_train = tf.placeholder(tf.bool,name='is_train')

        # 创建卷积层
        h1 = tf.layers.conv2d(tf_x_4d,filters=32,kernel_size=(3,3),activation=tf.nn.relu)
        max_h1 = tf.layers.max_pooling2d(h1,pool_size=(2,2),strides=(2,2),name='max_h1')
        h2 = tf.layers.conv2d(max_h1,filters=64,kernel_size=(2,2),activation=tf.nn.relu,name='h2')
        max_h2 = tf.layers.max_pooling2d(h2,pool_size=(2,2),strides=(2,2),name='max_h2')
        h3 = tf.layers.conv2d(max_h2,filters=128,kernel_size=(2,2),activation=tf.nn.relu,name='h3')
        max_h3 = tf.layers.max_pooling2d(h3,pool_size=(2,2),strides=(2,2),name='max_h3')
        # 创建全连接层
        input_shape = max_h3.get_shape().as_list()
        input_units = np.prod(input_shape[1:])
        max_h3_out = tf.reshape(max_h3,shape=(-1,input_units))
        h4 = tf.layers.dense(max_h3_out,units=500,activation=tf.nn.relu,name='h4')
        drop_h4 = tf.layers.dropout(h4,rate=self.drop_rete,training=is_train,name='drop_h4')
        h5 = tf.layers.dense(drop_h4,units=500,activation=tf.nn.relu,name='h5')
        # drop_h5 = tf.layers.dropout(h5,rate=self.drop_rete,training=is_train,name='drop_h5')
        # 构建输出层
        output = tf.layers.dense(h5,units=30,activation=None,name='output')

        # 构建成本函数
        cost = tf.add(tf.losses.mean_squared_error(tf_y,output),tf.constant(0,tf.float32),name='cost')
        # 构造优化器
        global_step = tf.Variable(0,trainable=False)
        lr = tf.train.exponential_decay(self.lr,global_step=global_step,decay_steps=100,decay_rate=0.9,name='lr')
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(cost,name='train_op')
        add_global = global_step.assign_add(1,name='add_global')
        # 预测
        prediction = tf.add(tf.multiply(output, tf.constant(48, dtype=tf.float32)), tf.constant(48, dtype=tf.float32),
                               name='prediction')

    def create_batch_generator(self,X,y):
        # 批数据生成器
        X_copy = np.copy(X)
        y_copy = np.copy(y)
        if self.shuffle:
            indices = np.random.shuffle(X_copy)
            X_copy = X_copy[indices]
            y_copy = y_copy[indices]
        for i in range(0,X_copy.shape[0],self.batch_size):
            yield (X_copy[i:i+self.batch_size,:],y_copy[i:i+self.batch_size,:])

    def data_augmentation(self,X,y,alter_rate=0.5):
        # 数据增强，默认图片数据以图中心为原点，反转后坐标x为求反
        indices = np.random.choice(X.shape[0],int(X.shape[0]*alter_rate),replace=False)
        flip_indices = [# 反转后需要变换的坐标,例如：左眼中心-》右眼中心
            (0, 2), (1, 3),
            (4, 8), (5, 9), (6, 10), (7, 11),
            (12, 16), (13, 17), (14, 18), (15, 19),
            (22, 24), (23, 25),
        ]
        X = np.reshape(X, (-1, 96, 96))
        tem_x = X[indices, :, ::-1]  # 翻转图片
        X = np.reshape(X, (-1, 96 * 96))
        tem_x = np.reshape(tem_x,(-1,96*96))
        if y is not None: # 交换坐标值
            tem_y = y[indices,::]
            tem_y[:,::2] = tem_y[:,::2]*-1
            for a,b in flip_indices:
                tem_a,tem_b = np.copy(tem_y[:,b]),np.copy(tem_y[:,a])
                tem_y[:,a],tem_y[:,b] = tem_a,tem_b
        # X = np.row_stack((X,tem_x))
        # y = np.row_stack((y,tem_y))
        X[indices,::] = tem_x
        y[indices,::] = tem_y
        return (X,y)



    def train(self,train_set,validation_set=None,initialize=True,data_augmentation=False):
        if initialize:
            self.sess.run(self.init_op)
        self.train_costs = []
        self.valid_costs = []
        for epoch in range(self.epoch):
            avg_cost = 0
            count = 0
            if data_augmentation:
                train_set = self.data_augmentation(train_set[0], train_set[1])
            batch_generator = self.create_batch_generator(train_set[0],train_set[1])
            for batch_x,batch_y in batch_generator:
                feed = {
                    'tf_x:0':batch_x,
                    'tf_y:0':batch_y,
                    'is_train:0':False
                }
                cost ,_ = self.sess.run(['cost:0','train_op'],feed_dict=feed)
                avg_cost += cost
                count += 1
            self.sess.run(['add_global'])
            self.train_costs.append(avg_cost/count)
            if validation_set:
                feed = {'tf_x:0':validation_set[0],'tf_y:0':validation_set[1],'is_train:0':False}
                valid_cost,lr = self.sess.run(['cost:0','lr:0'],feed_dict=feed)
                self.valid_costs.append(valid_cost)
                print('epoch-{} train_cost:{} valid_cosst:{} lr:{}'.format(epoch + 1, self.train_costs[-1], valid_cost,lr))
            else:
                print('epoch-{} train_cost:{}'.format(epoch + 1,self.train_costs[-1]))

    def save(self,save_path='./cnn-model/'):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print('saving model in %s' % save_path)
        self.saver.save(self.sess,save_path=os.path.join(save_path,'model.ckpt'),global_step=self.epoch)

    def load(self,save_path='./cnn-model/',epoch=400):
        print('loading model in %s' % save_path)
        self.saver.restore(self.sess,save_path=os.path.join(save_path,'model.ckpt-%d'%(epoch)))

    def predict(self,X):
        return cnn.sess.run(['prediction:0'], feed_dict={'tf_x:0': X, 'is_train:0': False})[0]


def create_submition(X_test,cnn):
    l = []
    [l.append(cnn.predict(X_test[x:x+1,:])) for x in range(X_test.shape[0])]
    l = np.array(l)
    l = l.flatten()
    l = np.reshape(l,(1783,30))
    l = pd.DataFrame(l,columns=['left_eye_center_x','left_eye_center_y','right_eye_center_x','right_eye_center_y','left_eye_inner_corner_x','left_eye_inner_corner_y','left_eye_outer_corner_x','left_eye_outer_corner_y','right_eye_inner_corner_x','right_eye_inner_corner_y','right_eye_outer_corner_x','right_eye_outer_corner_y','left_eyebrow_inner_end_x','left_eyebrow_inner_end_y','left_eyebrow_outer_end_x','left_eyebrow_outer_end_y','right_eyebrow_inner_end_x','right_eyebrow_inner_end_y','right_eyebrow_outer_end_x','right_eyebrow_outer_end_y','nose_tip_x','nose_tip_y','mouth_left_corner_x','mouth_left_corner_y','mouth_right_corner_x','mouth_right_corner_y','mouth_center_top_lip_x','mouth_center_top_lip_y','mouth_center_bottom_lip_x','mouth_center_bottom_lip_y']
)

    # 提交
    idlook = pd.read_csv('../../../machinelearndata/kaggle/facial/IdLookupTable.csv')
    locations = []
    for i in range(idlook.shape[0]):
        imageId = idlook.iloc[i]['ImageId']
        feature_name = idlook.iloc[i]['FeatureName']
        location = l.iloc[imageId-1][feature_name]
        locations.append(location)
    idlook['Location'] = locations
    idlook.to_csv('./submition.csv',columns=['RowId','Location'],index=False)

def show_img(y_train,X_train,cnn):
    y_pred = cnn.sess.run(['prediction:0'], feed_dict={'tf_x:0': X_train[0:16, :], 'is_train:0': False})[0]
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    y_train = np.multiply(y_train, 48)
    y_train = np.add(y_train, 48)
    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        img = X_test[i].reshape(96, 96)
        ax.imshow(img, cmap='gray')
        print(np.mean(np.square(y_train[i] - y_pred[i])))
        ax.scatter(y_pred[i][0::2], y_pred[i][1::2], marker='x', s=10)
        ax.scatter(y_train[i][0::2], y_train[i][1::2], marker='o', s=10, c='red')
    plt.show()





if __name__ == '__main__':
    cnn = CNN(batch_size=128,epoch=2000)
    # 分割数据集
    X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(X_train, y_train, test_size=0.2, random_state=123)
    cnn.train(train_set=(X_s_train,y_s_train),validation_set=(X_s_test,y_s_test),data_augmentation=True)
    cnn.save()
    # cnn.load(epoch=100)
    # show_img(y_train,X_train,cnn)
    create_submition(X_test,cnn)

