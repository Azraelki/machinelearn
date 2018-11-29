import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
'''
模型评估和参数调优
'''

# 在流水线中集成数据转换及评估操作
def function1():
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
                     header=None)

    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(le.classes_)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # 流水线
    pipe_lr = make_pipeline(StandardScaler(),PCA(n_components=2),LogisticRegression(random_state=1))
    pipe_lr.fit(X_train,y_train)
    y_pred = pipe_lr.predict(X_test)
    print('test accuracy: %.3f'%pipe_lr.score(X_test,y_test))

# 使用k-fold 交叉验证评估模型性能(使用StratifiedKFold迭代器实现)
def function2():
    from sklearn.model_selection import StratifiedKFold

    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
                     header=None)

    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(le.classes_)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    pipe_lr = make_pipeline(StandardScaler(),PCA(n_components=2),LogisticRegression(random_state=1))
    kfold = StratifiedKFold(n_splits=10,random_state=1).split(X_train,y_train)
    scores = []

    for k, (train,test) in enumerate(kfold):
        pipe_lr.fit(X_train[train],y_train[train])
        score = pipe_lr.score(X_train[test],y_train[test])
        scores.append(score)
        print("fold: %2d, class dist.: %s, acc: %.3f" % (k+1,np.bincount(y_train[train]),score))

# 使用k-fold 交叉验证对评估模型性能(cross_val_score实现)
def function3():
    from sklearn.model_selection import cross_val_score

    from sklearn.model_selection import StratifiedKFold

    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
                     header=None)

    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(le.classes_)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression(random_state=1))

    scores = cross_val_score(estimator=pipe_lr,
                             X=X_train,
                             y=y_train,
                             cv=10,# k的大小
                             n_jobs=1) # 使用cpu数，-1表示使用所有的

    print("CV accurancy scores: %s" % scores)
    print("CV accuracy: %.3f +/- %.3f" % (np.mean(scores),np.std(scores)))

##################通过学习及验证曲线来调试算法########################################

# 绘制学习曲线（准确率和样本数量的关系）
def function4():
    from sklearn.model_selection import learning_curve

    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
                     header=None)

    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(le.classes_)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2',random_state=1))

    train_sizes,train_scores,test_scores = learning_curve(estimator=pipe_lr,
                                                          X=X_train,
                                                          y=y_train,
                                                          train_sizes=np.linspace(0.1,1.0,10), # 每次迭代时所用样本的比例
                                                          cv=10, # 交叉验证的k值
                                                          n_jobs=1)
    train_mean = np.mean(train_scores,axis=1) # 列上求均值
    train_std = np.std(train_scores,axis=1) # 列上求标准差
    test_mean = np.mean(test_scores,axis=1)
    test_std = np.std(test_scores,axis=1)

    plt.plot(train_sizes,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')
    plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,
                     alpha=0.15,color='blue')

    plt.plot(train_sizes,test_mean,color='green',linestyle='--',marker='s',markersize=5,label='validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std,
                     color='green', alpha=0.15)

    plt.grid()
    plt.xlabel('number of training samples')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.8,1.0])
    plt.show()


# 绘制验证曲线（准确率和模型参数之间的关系）
def function5():
    from sklearn.model_selection import validation_curve

    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
                     header=None)

    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(le.classes_)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', random_state=1))

    param_range = [0.001,0.01,0.1,1.0,10.0,100.0] # 评估器参数C的范围

    train_scores, test_scores = validation_curve(estimator=pipe_lr,
                                                 X=X_train,
                                                 y=y_train,
                                                 param_name='logisticregression__C',
                                                 param_range=param_range,
                                                 cv=10)
    train_mean = np.mean(train_scores, axis=1)  # 列上求均值
    train_std = np.std(train_scores, axis=1)  # 列上求标准差
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5,
             label='validation accuracy')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std,
                     color='green', alpha=0.15)

    plt.grid()
    plt.xscale('log')
    plt.xlabel('parameter C')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.03])
    plt.show()


if __name__ == '__main__':
    # df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
    #                  header=None)
    #
    # X = df.loc[:,2:].values
    # y = df.loc[:,1].values
    # le = LabelEncoder()
    # y = le.fit_transform(y)
    # print(le.classes_)
    # X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

    # function1()
    # function2()
    # function3()
    # function4()
    function5()