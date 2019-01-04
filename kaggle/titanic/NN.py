import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import  LabelEncoder
import tensorflow.contrib.keras as keras
from sklearn.model_selection import train_test_split

data_raw  = pd.read_csv("../../../machinelearndata/kaggle/titanic/train.csv", encoding="utf-8")
data_val   = pd.read_csv("../../../machinelearndata/kaggle/titanic/test.csv", encoding="utf-8")

data1 = data_raw.copy(deep=True)
data_cleaner = [data1,data_val]
# 1、补充或者删除两个数据集的缺失值

for dataset in data_cleaner:
    # 使用中位数数补全Age
    dataset['Age'].fillna(dataset['Age'].median(),inplace=True)
    # 使用众数补全Embarked
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0],inplace=True)
    # 使用中位数补全Fare
    dataset['Fare'].fillna(dataset['Fare'].median(),inplace=True)

# 删除列
drop_column = ['PassengerId','Cabin','Ticket']
data1.drop(drop_column,axis=1,inplace=True)

print(data1.isnull().sum())
print('-'*10)
print(data_val.isnull().sum())

# 2、特征工程
for dataset in data_cleaner:
    # 创建家庭字段
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 1
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0

    # 获取名字的title
    dataset['Title'] = dataset['Name'].str.split(", ",expand=True)[1].str.split(".",expand=True)[0]

    # 将Fare和Age分段
    dataset['FareBin'] = pd.qcut(dataset['Fare'],4)
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int),5)

# 清理稀有的title
print(data1['Title'].value_counts())
stat_min = 10 # 出现最小次数
title_names = (data1['Title'].value_counts()<stat_min)
data1['Title'] = data1['Title'].apply(lambda x:'Misc' if title_names.loc[x] else x)
print(data1['Title'].value_counts())
print("-"*10)

print(data1.info())
print(data_val.info())
print(data1.sample(10))

##3、 转换格式
#
label = LabelEncoder()
for dataset in data_cleaner:
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])

# 定义 y 量
Target = ['Survived']
# 定义x变量
data1_x = ['Sex','Pclass','Embarked','Title','SibSp','Parch','Age','Fare',
           'FamilySize','IsAlone']
data1_x_clac = ['Sex_Code','Pclass','Embarked_Code','Title_Code','SibSp','Parch',
                'Age','Fare']

data1_xy = Target+data1_x
print('original x y:',data1_xy,'\n')

data1_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
data1_xy_bin = Target + data1_x_bin
print('bin x y',data1_xy_bin,'\n')

data1_dummy = pd.get_dummies(data1[data1_x])
data1_x_dummy = data1_dummy.columns.tolist()
data1_xy_dummy = Target + data1_x_dummy
print('dummy x y:',data1_xy_dummy,'\n')

train1_x_bin,test1_x_bin,train1_y_bin,test1_y_bin = train_test_split(data1[data1_x_bin],data1[Target],random_state=0,test_size=0.3)

# 构建计算图
g = tf.Graph()
with g.as_default():
    tf.set_random_seed(123)
    # 定义输入变量占位符X,y
    tf_x = tf.placeholder(tf.float32,shape=(None,7),name='tf_x')
    tf_y = tf.placeholder(tf.int32,shape=(None),name='tf_y')
    #定义隐藏层-1
    h1 = tf.layers.dense(inputs=tf_x,units=1000,activation=tf.tanh,name='layer1')
    h2 = tf.layers.dense(inputs=h1,units=1000,activation=tf.tanh,name='layer2')
    # 定义输出层
    logits = tf.layers.dense(inputs=h2,units=1,activation=tf.sigmoid,name='logits')

    # 定义成本函数
    cost = tf.losses.sigmoid_cross_entropy(tf_y,logits=logits)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

    predictions = {
        'classes': tf.round(logits)
    }

    train_op = optimizer.minimize(cost)
    init_op = tf.global_variables_initializer()

# 开启一个会话
with tf.Session(graph=g) as sess:
    sess.run(init_op)
    # 迭代训练100次

    for epoch in range(100):
        train_costs = []
        feed = {tf_x: train1_x_bin.values.astype(np.float32), tf_y: train1_y_bin.values}
        _, costs= sess.run([train_op, cost], feed_dict=feed)
        train_costs.append(costs)
        print('-- epoch %2d avg training loss: %.4f' % (epoch + 1, np.mean(train_costs)))

    feed = {tf_x:test1_x_bin}
    y_pred = sess.run(predictions['classes'],feed_dict=feed)
    y_pred = y_pred.flatten()
    y_test = test1_y_bin.values.flatten()
    print("test accuracy:%.2f%%"%(100*np.sum(y_pred==y_test)/len(test1_y_bin)))
    pass



