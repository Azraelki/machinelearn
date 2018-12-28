'''
kaggle竞赛-----tatinic人员幸存预测
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.metrics import roc_curve
from sklearn.model_selection import ShuffleSplit

import os


# 分析数据-1
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
    # Cabin             204          91     缺失较多          观察数据，此字段为房间号，因缺失较多，考虑是否缺失值代表了某种信息
    # Fare              891         417     极少缺失          定量变量，暂时使用均值代替
    # Embarked          889         418     极少缺失          定性变量，暂时使用众数代替

    # 处理数据集-1
    train_data['Age'] = train_data['Age'].fillna(train_data.loc[:, 'Age'].mean())
    test_data['Age'] = test_data['Age'].fillna(train_data.loc[:, 'Age'].mean())
    train_data['Embarked'] = train_data['Embarked'].fillna('S')
    test_data['Embarked'] = test_data['Embarked'].fillna('S')
    train_data['Fare'] = train_data['Fare'].fillna(train_data.loc[:,'Fare'].mean())
    test_data['Fare'] = test_data['Fare'].fillna(train_data.loc[:,'Fare'].mean())
    train_data = train_data.drop(columns=['PassengerId','Ticket'])
    test_data = test_data.drop(columns=['PassengerId','Ticket'])
    # train_data['Cabin'] = train_data['Cabin'][]
    # test_data['Cabin'] = test_data.drop(columns=['PassengerId', 'Ticket'])
    print(train_data.shape,test_data.shape)

    ## 字段分析
    # PassengerId	int 类型，计数使用，直接丢弃
    # Survived      {0,1}，幸存与否，此字段作为类标签
    # Pclass        {1,2,3}，船票种类，根据图-1的图像分析，可以得出，不同船票种类的生存概率： p(Survived|1) > p(Survived|2) > p(Survived|3)
    # Name          字符串类型，初步先不对其做分析，根据其名称的不同有可能得出一些信息
    # Sex           {male，female}，性别，根据图-2分析，女性的生存概率要大于男性
    # Age            int,年龄，由图-3第二幅图分析得出，四个年龄阶段中child的幸存比例明显较高，其他几个年龄阶段幸存大致相同
    # SibSp          int,子侄关系数量，通过图-4分析，关系数越少生存概率越大，SibSp和Parch分布趋势高度一致，可考虑合并这两项为一个特征
    # Parch          int,父母子女关系数量，通过图-4分析，关系数越少生存概率越大，SibSp和Parch分布趋势高度一致，可考虑合并这两项为一个特征
    # Ticket         每个数据样本的值都不太一样，暂时先丢弃此列
    # Fare           float,船票费用，根据图-5分析，船票费用越高幸存几率越大
    # Cabin          字母序列，船舱号，根据图-6分析，有船舱号的人幸存几率大
    # Embarked       字母序列，登船地，根据图-7分析，不同登船地的幸存概率为 p(Survived|C) > p(Survived|Q) > p(Survived|S)

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

    # 图-4 SibSp和Parch与Survived的关系
    df = train_data[['SibSp', 'Parch', 'Survived']]
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122, sharey=ax1)
    sns.barplot(x='SibSp', y='Survived', data=df, estimator=np.sum, ax=ax1)
    sns.barplot(x='Parch', y='Survived', data=df, estimator=np.sum, ax=ax2)
    ax1.set_title("SibSp-Survived Bar graph")
    ax2.set_title("Parch-Survived Bar graph")
    plt.show()

    # 图-5 Fare-Survuved关系
    df = train_data[['Fare', 'Survived']]
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    sns.scatterplot(x='Fare', y='Survived', hue='Survived', data=df, ax=ax1)
    media_num = train_data['Fare'].median()
    df1 = df[df['Fare'] < media_num]['Survived']
    df2 = df[df['Fare'] >= media_num]['Survived']
    ax2.pie([df1.sum(), df2.sum()], labels=['low fare', 'high fare'],
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.show()

    # 图-6 Cabin-Survived关系图
    df = train_data[['Cabin', 'Survived']]
    count1 = df['Cabin'].count() - df[df['Cabin'].isna()]['Cabin'].count()  # 有船舱号的数量
    count2 = df['Survived'].sum() - df[df['Cabin'].isna()]['Survived'].sum()  # 有船舱号的幸存者数量
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.pie([count1, df['Survived'].count() - count1], labels=['has cabin', 'no cabin'],
            autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.set_title('human rate')
    ax2.pie([count2 * 10 / count1, (df['Survived'].sum() - count2) * 10 / (df['Survived'].count() - count1)],
            labels=['has cabin', 'no cabin'],
            autopct='%1.1f%%', shadow=True, startangle=90)
    ax2.set_title('survived rate')
    plt.show()

    # 图-7 Embarked-Survived 关系图
    df = train_data[['Embarked', 'Survived']]
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.pie([df[df['Embarked'] == 'S']['Embarked'].count(), df[df['Embarked'] == 'Q']['Embarked'].count(),
             df[df['Embarked'] == 'C']['Embarked'].count()],
            labels=['S', 'Q', 'C'], autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.set_title('human rate')
    ax2.pie([df[df['Embarked'] == 'S']['Survived'].sum() / df[df['Embarked'] == 'S']['Embarked'].count(),
             df[df['Embarked'] == 'Q']['Survived'].sum() / df[df['Embarked'] == 'Q']['Embarked'].count(),
             df[df['Embarked'] == 'C']['Survived'].sum() / df[df['Embarked'] == 'C']['Embarked'].count()],
            labels=['S', 'Q', 'C'], autopct='%1.1f%%', shadow=True, startangle=90)
    ax2.set_title('survived rate')
    plt.show()


# 处理数据-1
def deal_data_1():
    # 获取数据集
    train_data = pd.read_csv("../../../machinelearndata/kaggle/titanic/train.csv", encoding="utf-8")
    test_data = pd.read_csv("../../../machinelearndata/kaggle/titanic/test.csv", encoding="utf-8")

    # 根据analize_data_1()函数中的分析处理数据
    # 均值填充Age缺失字段
    train_data['Age'] = train_data['Age'].fillna(train_data.loc[:, 'Age'].mean())
    test_data['Age'] = test_data['Age'].fillna(train_data.loc[:, 'Age'].mean())
    # 使用众数填充登船字段的缺失值
    train_data['Embarked'] = train_data['Embarked'].fillna('S')
    test_data['Embarked'] = test_data['Embarked'].fillna('S')
    # 使用均值填充船票费用
    train_data['Fare'] = train_data['Fare'].fillna(train_data.loc[:, 'Fare'].mean())
    test_data['Fare'] = test_data['Fare'].fillna(train_data.loc[:, 'Fare'].mean())
    # 丢弃 Name Ticket PassengerId三个字段
    train_data = train_data.drop(columns=['PassengerId', 'Ticket','Name'])
    test_data = test_data.drop(columns=['Ticket','Name'])

    # 将Sex字段修改为数值型，1：male 0:female
    train_data['Sex'] = train_data['Sex'].replace(['male','female'],[1,0])
    test_data['Sex'] = test_data['Sex'].replace(['male','female'],[1,0])

    # 将Cabin字段修改为数值类型 1：有值 0：nan
    train_data['Cabin'] = train_data['Cabin'].fillna(0)
    train_data.Cabin[train_data['Cabin']!=0] = 1
    test_data['Cabin'] = test_data['Cabin'].fillna(0)
    test_data.Cabin[test_data['Cabin'] != 0] = 1

    # 将Embarked修改为数值型数值 1：S 2：Q 3：C
    train_data['Embarked'] = train_data['Embarked'].replace(['S','Q','C'],[1,2,3])
    test_data['Embarked'] = test_data['Embarked'].replace(['S','Q','C'],[1,2,3])

    # 将数据转换为np数组
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    # 将Age和Fare进行归一化
    std_age = StandardScaler()
    train_data[:,3] = std_age.fit_transform(train_data[:,3].reshape(-1,1)).reshape(len(train_data[:,3]))
    test_data[:,3] = std_age.transform(test_data[:,3].reshape(-1,1)).reshape(len(test_data[:,3]))

    std_fare = StandardScaler()
    train_data[:, 6] = std_fare.fit_transform(train_data[:, 6].reshape(-1, 1)).reshape(len(train_data[:, 6]))
    test_data[:, 6] = std_fare.transform(test_data[:, 6].reshape(-1, 1)).reshape(len(test_data[:, 6]))

    # SibSp和Parch两个可以进行合并为一个属性--家庭关系属性 family
    train_data = np.column_stack((train_data,train_data[:,4]+train_data[:,5]))
    train_data = np.delete(train_data,[4,5],axis=1)

    test_data = np.column_stack((test_data, test_data[:, 4] + test_data[:, 5]))
    test_data = np.delete(test_data, [4, 5], axis=1)

    # 将Embarked one-hot化，因为登陆地两两之间关联度是一致
    onehot = OneHotEncoder(categories='auto')
    onehot.fit(train_data[:,-2].reshape(-1,1))
    train_data = np.column_stack((train_data,onehot.transform(train_data[:,-2].reshape(-1,1)).toarray()))
    test_data = np.column_stack((test_data,onehot.transform(test_data[:,-2].reshape(-1,1)).toarray()))
    train_data = np.delete(train_data, [-5], axis=1)
    test_data = np.delete(test_data, [-5], axis=1)

    # 处理后各列数据的含义
    # train_data:['Survived','Pclass','Sex','Age','Fare','Cabin','Family','Embarked_S','Embarked_Q','Embarked_C']
    # test_data: ['Pclass','Sex','Age','Fare','Cabin','Family','Embarked_S','Embarked_Q','Embarked_C']
    return train_data,test_data # 此处返回清理过后的训练数据集和最终提交的数据集

# 训练模型-1
def train_model_1():
    np.random.seed(123)
    # 获取处理后的数据
    train,test = deal_data_1()
    print(train.shape,test.shape)
    # 划分训练和测试数据集
    X = train[:,1:].astype('float')
    y = train[:,0].astype('int')
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=123,stratify=y)
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

    # 构建一个逻辑斯谛回归模型
    logist = LogisticRegression(penalty='l2',C=1.0,random_state=123)
    logist.fit(X_train,y_train)
    print("training accuracy: ", logist.score(X_train, y_train))
    print("test accuracy: ", logist.score(X_test, y_test))
    print("coef:",logist.coef_)

    # 构建一个GBDT（梯度提升树）PS:此处的数据中Embarked可以不需要进行one-hot化
    gbdt = GradientBoostingClassifier(learning_rate=0.01,n_estimators=800,subsample=0.7,random_state=123)
    gbdt.fit(X_train,y_train)
    print("training accuracy: ", gbdt.score(X_train, y_train))
    print("test accuracy: ", gbdt.score(X_test, y_test))
    print("future importance:",gbdt.feature_importances_)

    # 使用交叉验证进行模型评估
    scores_logist = cross_val_score(logist,X,y,cv=10)
    scores_gbdt = cross_val_score(gbdt,X,y,cv=10)
    print("logist(mean) cross_val_score:",scores_logist.mean())
    print("gbdt(mean) cross_val_score:",scores_gbdt.mean())

    # logist 使用网格搜索调优参数
    param_grid = [
        {
            'C':[10**i for i in range(-5,5)],
            'penalty':['l1','l2']
        }
    ]
    gs_logist = GridSearchCV(estimator=logist,param_grid=param_grid,scoring='accuracy',verbose=1)
    gs_logist.fit(X_train, y_train)

    print('best parameter set:%s' % gs_logist.best_params_) # {C:1 penalty:l2}

    print('cv accuracy: %.3f' % gs_logist.best_score_)

    clf_logist = gs_logist.best_estimator_
    print('test accuracy: %.3f' % clf_logist.score(X_test, y_test))

    # GBDT 使用网格搜索调优参数
    param_grid = [
        {
            'learning_rate': [0.002,0.004,0.008,0.01,0.02,0.04,0.08,0.1,0.2,0.3,0.4],
            'n_estimators': [200,400,600,800,1000],
            'subsample':[0.7,0.8,0.9,1.0]
        }
    ]
    gs_gbdt = GridSearchCV(estimator=gbdt, param_grid=param_grid, scoring='accuracy', verbose=5)
    gs_gbdt.fit(X_train, y_train)

    print('best parameter set:%s' % gs_gbdt.best_params_) # {learning_rate:0.01,n_estimators:800,subsample:0.7}

    print('cv accuracy: %.3f' % gs_gbdt.best_score_)

    clf_gbdt = gs_gbdt.best_estimator_
    print('test accuracy: %.3f' % clf_gbdt.score(X_test, y_test))

# 绘制学习曲线
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=ShuffleSplit(n_splits=100,test_size=0.2,random_state=123),
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,verbose=10)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# 导出结果，并绘制学习曲线
def train_model():
    np.random.seed(123)
    # 获取处理后的数据
    train, test = deal_data_1()
    print(train.shape, test.shape)
    # 划分训练和测试数据集
    X = train[:, 1:].astype('float')
    y = train[:, 0].astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # 构建一个逻辑斯谛回归模型
    logist = LogisticRegression(penalty='l2', C=1.0, random_state=123)
    logist.fit(X_train, y_train)
    print("training accuracy: ", logist.score(X_train, y_train))
    print("test accuracy: ", logist.score(X_test, y_test))
    print("coef:", [i for i in zip(['Pclass','Sex','Age','Fare','Cabin','Family','Embarked_S','Embarked_Q','Embarked_C'],logist.coef_.flatten())])
    # 导出输出结果
    predictions = logist.predict(test[:,1:].astype('float'))
    result = pd.DataFrame({'PassengerId': test[:,0], 'Survived': predictions.astype(np.int32)})
    result.to_csv("./logistic_regression_predictions.csv", index=False)

    # 构建一个GBDT（梯度提升树）PS:此处的数据中Embarked可以不需要进行one-hot化
    gbdt = GradientBoostingClassifier(learning_rate=0.01, n_estimators=800, subsample=0.7, random_state=123)
    gbdt.fit(X_train, y_train)
    print("training accuracy: ", gbdt.score(X_train, y_train))
    print("test accuracy: ", gbdt.score(X_test, y_test))
    print("future importance:", sorted([ i for i in zip(['Pclass','Sex','Age','Fare','Cabin','Family','Embarked_S','Embarked_Q','Embarked_C'],gbdt.feature_importances_)],key=lambda x:x[1],reverse=True))
    # 导出输出结果
    predictions = gbdt.predict(test[:,1:].astype('float'))
    result = pd.DataFrame({'PassengerId': test[:,0], 'Survived': predictions.astype(np.int32)})
    result.to_csv("./gbdt_predictions.csv", index=False)


    # [('Pclass', -0.6389829630001616), ('Sex', -2.6953339025829166), ('Age', -0.5179257067078015), ('Fare', 0.09360925029269106), ('Cabin', 0.8366997509506894), ('Family', 0.713749563898081), ('Embarked_S', -0.18383638046527376), ('Embarked_Q', 0.8794868668069129), ('Embarked_C', 0.2158091452296257)]
    # [('Sex', 0.4085), ('Fare', 0.2013), ('Age', 0.1825), ('Pclass', 0.0784), ('Embarked_S', 0.0662), ('Cabin', 0.0356), ('Family', 0.0106), ('Embarked_Q', 0.0090), ('Embarked_C', 0.0036)]
    # logistic的参数和GBDT生成的特征重要性 与analize_data_1中的分析进行对比
    # 1、Pclass、Sex、Age、Fare与我们对数据的分析几乎一致，表现出了预期的相关性
    # 2、Cabin字段重要性排第5而logistic系数表现为很强的正相关，此字段是否需要再深入挖掘下
    # 3、Embarked字段，并没有达到预期的效果，Embarked_Q和Embarked_C虽然有大的正相关系数，但是在GBDT中显示其重要性很低，只是明显显示了Embarked_S会拉低生存率
    # 4、Family字段与我们预期的正好相反，其表现为正相关，与我们预期的值越小生存率越低不一致


    # 绘制学习曲线，观察拟合情况(普通电脑画出图形需要一段时间，根据电脑配置不同时间不定)
    plot_learning_curve(estimator=logist,title='logistic learning curve',X=X,y=y)
    plot_learning_curve(estimator=gbdt,title='gbdt learning curve',X=X,y=y,n_jobs=2)
    plt.show()

##################################以上完成了第一次的提交，两个模型的得分都在0.7703#####################################################

#################################################以下为第二次分析####################################################################
# 交叉验证
def across_validation(estimitor=None,title="estimitor",X=None,y=None,cv=10):
    scores = cross_val_score(estimitor, X, y, cv=10)
    print("%s  cross_val_score: min:%.4f  max:%.4f  mean:%.4f"%(title,np.min(scores),np.max(scores),scores.mean()))
# 分析数据-2   特征工程
def analize_data_2():
    # 获取数据集
    train_data_origin = pd.read_csv("../../../machinelearndata/kaggle/titanic/train.csv", encoding="utf-8")
    test_data_origin = pd.read_csv("../../../machinelearndata/kaggle/titanic/test.csv", encoding="utf-8")

    # 处理-1
    train_data = train_data_origin.copy()
    test_data = test_data_origin.copy()
    train_data['Age'] = train_data['Age'].fillna(train_data.loc[:, 'Age'].mean())
    test_data['Age'] = test_data['Age'].fillna(train_data.loc[:, 'Age'].mean())
    train_data['Embarked'] = train_data['Embarked'].fillna('S')
    test_data['Embarked'] = test_data['Embarked'].fillna('S')
    train_data['Fare'] = train_data['Fare'].fillna(train_data.loc[:, 'Fare'].mean())
    test_data['Fare'] = test_data['Fare'].fillna(train_data.loc[:, 'Fare'].mean())
    sex_map = {'male': 1, 'female': 0}
    train_data['Sex'] = train_data['Sex'].map(sex_map)
    test_data['Sex'] = test_data['Sex'].map(sex_map)
    embarked_map = {'S': 1, 'Q': 2,'C':3}
    train_data['Embarked'] = train_data['Embarked'].map(embarked_map)
    test_data['Embarked'] = test_data['Embarked'].map(embarked_map)
    train_data['Cabin'] = train_data['Cabin'].fillna(0)
    train_data.Cabin[train_data['Cabin'] != 0] = 1
    test_data['Cabin'] = test_data['Cabin'].fillna(0)
    test_data.Cabin[test_data['Cabin'] != 0] = 1


    logist = LogisticRegression(penalty='l2', C=1.0, random_state=123,solver='lbfgs',max_iter=500)
    gbdt = GradientBoostingClassifier(learning_rate=0.01, n_estimators=800, subsample=0.7, random_state=123)
    print("###################处理----1######################")
    # across_validation(estimitor=logist,title='logistic',X=train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']].values,y=train_data['Survived'].values)
    # across_validation(estimitor=gbdt,title='gbdt',X=train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']].values,y=train_data['Survived'].values)
    # logistic  cross_val_score: min:0.7640  max:0.8222  mean:0.7957
    # gbdt  cross_val_score: min:0.7753  max:0.9101  mean:0.8329

    # 处理-2
    # 尝试将Embarked除掉（经第一次的模型结果分析，Embarked字段重要性很低，并且出现了S严重拉低生存率的现象）
    logist = LogisticRegression(penalty='l2', C=1.0, random_state=123,solver='lbfgs',max_iter=500)
    gbdt = GradientBoostingClassifier(learning_rate=0.01, n_estimators=800, subsample=0.7, random_state=123)
    print("###################处理----2######################")
    # across_validation(estimitor=logist, title='logistic',
    #                   X=train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']].values,
    #                   y=train_data['Survived'].values)
    # across_validation(estimitor=gbdt, title='gbdt',
    #                   X=train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin',]].values,
    #                   y=train_data['Survived'].values)
    # logistic  cross_val_score: min:0.7640  max:0.8315  mean:0.7946
    # gbdt  cross_val_score: min:0.7753  max:0.8989  mean:0.8362
    # 与处理-1相比略微有提升

    # 处理-3
    # 修改age的空值填充方式，可以从Name的称呼上进行分组划分估计,名字中包含的Mr,Mrs,Miss,Dr和Master五个类型
    train_data['Age'] = train_data_origin['Age']
    mean_mr = train_data[train_data['Name'].str.contains('Mr\.')]['Age'].dropna().mean()
    mean_mrs = train_data[train_data['Name'].str.contains('Mrs\.')]['Age'].dropna().mean()
    mean_miss = train_data[train_data['Name'].str.contains('Miss\.')]['Age'].dropna().mean()
    mean_master = train_data[train_data['Name'].str.contains('Master\.')]['Age'].dropna().mean()
    mean_dr = train_data[train_data['Name'].str.contains('Dr\.')]['Age'].dropna().mean()
    train_data.loc[train_data['Name'].str.contains('Mr\.'),'Age'] = mean_mr
    train_data.loc[train_data['Name'].str.contains('Mrs\.'), 'Age'] = mean_mrs
    train_data.loc[train_data['Name'].str.contains('Miss\.'), 'Age'] = mean_miss
    train_data.loc[train_data['Name'].str.contains('Master\.'), 'Age']= mean_master
    train_data.loc[train_data['Name'].str.contains('Dr\.'), 'Age']= mean_master
    # 归一化年龄
    std_age = StandardScaler()
    std_age.fit(train_data['Age'].values.reshape(-1,1))
    train_data['Age'] = std_age.transform(train_data['Age'].values.reshape(-1,1))
    # 归一化费用
    std_fare = StandardScaler()
    std_age.fit(train_data['Fare'].values.reshape(-1, 1))
    train_data['Fare'] = std_age.transform(train_data['Fare'].values.reshape(-1, 1))

    logist = LogisticRegression(penalty='l2', C=1.0, random_state=123, solver='lbfgs', max_iter=500)
    gbdt = GradientBoostingClassifier(learning_rate=0.01, n_estimators=800, subsample=0.7, random_state=123)
    print("###################处理----3######################")
    across_validation(estimitor=logist, title='logistic',
                      X=train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']].values,
                      y=train_data['Survived'].values)
    across_validation(estimitor=gbdt, title='gbdt',
                      X=train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', ]].values,
                      y=train_data['Survived'].values)
    # logistic  cross_val_score: min:0.7528  max:0.8315  mean:0.7957
    # gbdt  cross_val_score:  min:0.7640  max:0.8876  mean:0.8384
    # 与处理-2相比略微有提升


    # ## Fare字段挖掘分析                      count    mean     std    min     25%     50%     75%
    # print(train_data['Fare'].describe()) #     891    32.20   49.69    0      7.91   14.45     31
    # print(test_data['Fare'].describe())  #     417    35.62   55.90    0      7.89   14.45     31.5   (缺失一个)
    # # 对Fare分段处理,简单观察原始数据后分组  (1,0=<Fare<7.925) (2,7.925=<Fare<26.25) (3,26.25=<Fare)
    # df = train_data[train_data['Sex']==1]
    # df.sort_values(by=['Fare'],inplace=True,na_position='first')
    # # 图-8 对Fare挖掘绘制图

    pass

def deal_data_2(train_data):
    # 获取数据集
    train_data_origin = pd.read_csv("../../../machinelearndata/kaggle/titanic/train.csv", encoding="utf-8")
    test_data_origin = pd.read_csv("../../../machinelearndata/kaggle/titanic/test.csv", encoding="utf-8")



if __name__ == '__main__':
    # get_data()
    # analize_data_1()
    # deal_data_1()
    # train_model_1()
    # train_model()
    # feature_analyzie()
    analize_data_2()