'''
kaggle竞赛-----tatinic人员幸存预测
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import os

# 获取数据
def get_data(show=True):
    train_data = pd.read_csv("../../../machinelearndata/kaggle/titanic/train.csv", encoding="utf-8")
    test_data = pd.read_csv("../../../machinelearndata/kaggle/titanic/test.csv", encoding="utf-8")

    ### 清洗训练数据-begin
    # 统计存在空值的列
    print(train_data.count(),train_data.shape) ## 由打印结果可知age和cabin,Embarked两列有数据缺失

    # 使用均值填充Age列，使用中位数填充Embarked,丢弃Cabin列
    train_data['Age'] = train_data['Age'].fillna(train_data.loc[:,'Age'].mean())
    test_data['Age'] = test_data['Age'].fillna(test_data.loc[:,'Age'].mean())

    print(train_data['Embarked'].describe()) # 打印Embarked列的描述，得知最多的S
    train_data['Embarked'] = train_data['Embarked'].fillna('S')
    test_data['Embarked'] = test_data['Embarked'].fillna('S')

    train_data = train_data.dropna(axis=1) # 删除Cabin列
    test_data = test_data.dropna(axis=1) # 删除Cabin列

    train_data = train_data.drop(columns=['PassengerId','Name','Ticket']) # 删除Name和Ticket列
    test_data = test_data.drop(columns=['PassengerId','Name','Ticket']) # 删除Name和Ticket列
    print(train_data.count(),train_data.shape)

    # 将定性变量转换为数值型数据
    train_data['Sex'] = train_data['Sex'].replace(['male','female'],[1,0])
    test_data['Sex'] = test_data['Sex'].replace(['male','female'],[1,0])
    train_data['Embarked'] = train_data['Embarked'].replace(['C','S','Q'],[1,2,3])
    test_data['Embarked'] = test_data['Embarked'].replace(['C','S','Q'],[1,2,3])


    if show:
        ## 根据train_data保留的特征画出散列图观察
        sns.set(style='whitegrid',context='notebook')
        sns.pairplot(train_data,size=1.)
        plt.show()

    return train_data,test_data

def analize_data_1():
    # 获取数据集
    train_data = pd.read_csv("../../../machinelearndata/kaggle/titanic/train.csv", encoding="utf-8")
    test_data = pd.read_csv("../../../machinelearndata/kaggle/titanic/test.csv", encoding="utf-8")

    # 查看数据集的整体情况
    print("train数据集的情况汇总：")
    print(train_data.info())
    print("test数据集的情况汇总：")
    print(test_data.info())
    ## 通过输出的信息可以看出数据的缺失情况,总共四个属性有缺失值
    #                 train        test     描述              初步处理意见
    # 总条数            891         418
    # Age               714         332     缺失情况一般      定量变量，可以先用均值代替
    # Cabin             204          91     缺失较多          观察数据，此字段为房间号，发现其取值太过分散，可以考虑先放弃此属性
    # Fare              891         417     极少缺失          定量变量，暂时使用均值代替
    # Embarked          889         418     极少缺失          定性变量，暂时使用众数代替

    # 处理数据集-1
    train_data['Age'] = train_data['Age'].fillna(train_data.loc[:, 'Age'].mean())
    test_data['Age'] = test_data['Age'].fillna(train_data.loc[:, 'Age'].mean())
    train_data['Embarked'] = train_data['Embarked'].fillna('S')
    test_data['Embarked'] = test_data['Embarked'].fillna('S')
    train_data['Fare'] = train_data['Fare'].fillna(train_data.loc[:,'Fare'].mean())
    test_data['Fare'] = test_data['Fare'].fillna(train_data.loc[:,'Fare'].mean())
    train_data = train_data.drop(columns=['Cabin','PassengerId'])
    test_data = test_data.drop(columns=['Cabin','PassengerId'])
    print(train_data.shape,test_data.shape)

    ## 字段分析
    # PassengerId	int 类型，计数使用，直接丢弃
    # Survived      {0,1}，幸存与否，此字段作为类标签
    # Pclass        {1,2,3}，船票种类，根据图-1的图像分析，可以得出，不同船票种类的生存概率： p(Survived|1) > p(Survived|2) > p(Survived|3)
    # Name          字符串类型，初步先不对其做分析，根据其名称的不同有可能得出一些信息
    # Sex           {male，female}，性别，根据图-2分析，女性的生存概率要大于男性
    # Age            int,年龄，
    # SibSp
    # Parch
    # Ticket
    # Fare
    # Cabin
    # Embarked

    # 图-1 Pclass与Survived的关系
    df = train_data[['Pclass','Survived']]
    Pclass_unique = train_data['Pclass'].unique()
    df1 = df[df['Survived']==1] # 存活的
    df2 = df[df['Survived']==0] # 未存活的
    Survived_count = [ df1[df1['Pclass'] == i]['Survived'].count() for i in Pclass_unique ]
    not_Survived_count = [ df2[df2['Pclass'] == i]['Survived'].count() for i in Pclass_unique ]
    plt.bar(Pclass_unique,np.divide(Survived_count,np.add(Survived_count,not_Survived_count)),facecolor='green',label='Survived') # 绘制不同类别船票的生存概率
    plt.xticks(Pclass_unique)
    plt.xlabel("Pclass")
    plt.ylabel("Survived rate (%)")
    plt.show()

    # 图-2 Sex与Survived的关系
    df = train_data[['Sex', 'Survived']]
    male = df[df['Sex']=='male']
    female = df[df['Sex']=='female']
    fig,ax = plt.subplots(nrows=1,ncols=2)
    ax[0].pie([male['Sex'].count(),female['Sex'].count()],labels=['male','female'],
              autopct='%1.1f%%',shadow=True,startangle=90)
    ax[1].pie([male[male['Survived']==1]['Survived'].sum(),
             female[female['Survived']==1]['Survived'].sum()],
            labels=['male','female'],autopct='%1.1f%%',shadow=True,startangle=90)
    ax[0].set_title("Sex rate")
    ax[1].set_title("Survied rate")
    ax[0].legend(loc='upper center',)
    ax[1].legend(loc='upper center')
    plt.show()

    # 图-3 Age与Survived的关系,可以按年龄段进行分组
    df = train_data[['Age', 'Survived']]
    df_1 = df[df['Age'] <= 14]['Survived'] # 年幼的
    df = df[df['Age'] > 14]
    df_2 = df[df['Age'] <= 24]['Survived'] # 青少年
    df = df[df['Age'] > 24]
    df_3 = df[df['Age'] <= 50]['Survived'] # 壮年
    df_4 = df[df['Age'] > 50]['Survived']
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    # 年龄段比例
    ax1.pie([df_1.count(),df_2.count(),df_3.count(),df_4.count()], labels=['child', 'teenager','adult','old'],
              autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.set_title("age rate")
    # 各个年龄段幸存比例
    ax2.pie([df_1.sum()/df_1.count(), df_2.sum()/df_2.count(), df_3.sum()/df_3.count(), df_4.sum()/df_4.count()], labels=['child', 'teenager', 'adult', 'old'],
            autopct='%1.1f%%', shadow=True, startangle=90)
    ax2.set_title("survived rate")
    plt.show()








    pass



if __name__ == '__main__':
    # get_data()
    analize_data_1()