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

##################使用网格搜索调优机器学习模型########################################
# 使用网格搜索调优超参
def function6():
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC

    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
                     header=None)

    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(le.classes_)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    pipe_svc = make_pipeline(StandardScaler(),SVC(random_state=1))

    param_range = [0.0001,0.001,0.01,0.1,1,10.0,100.0,1000.0]
    param_grid = [
        {
            'svc__C':param_range,
            'svc__kernel':['linear'] # 线性svm只需要调优C参数
        },{
            'svc__C':param_range,
            'svc__gamma':param_range,
            'svc__kernel':['rbf'] # 基于rbf的svm需要调优C参数和gamma参数
        }
    ]

    gs = GridSearchCV(estimator=pipe_svc,
                      param_grid=param_grid,
                      scoring='accuracy', # 评分标准
                      cv=10,
                      n_jobs=-1)

    gs = gs.fit(X_train,y_train)
    print(gs.best_score_)
    print(gs.best_params_)

# 使用嵌套交叉验证选择模型算法
def function7():
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import cross_val_score
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
                     header=None)

    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(le.classes_)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # svm模型嵌套交叉验证
    pipe_svc = make_pipeline(StandardScaler(),SVC(random_state=1))

    param_range = [0.0001,0.001,0.01,0.1,1,10.0,100.0,1000.0]
    param_grid = [
        {
            'svc__C':param_range,
            'svc__kernel':['linear'] # 线性svm只需要调优C参数
        },{
            'svc__C':param_range,
            'svc__gamma':param_range,
            'svc__kernel':['rbf'] # 基于rbf的svm需要调优C参数和gamma参数
        }
    ]

    gs = GridSearchCV(estimator=pipe_svc,
                      param_grid=param_grid,
                      scoring='accuracy', # 评分标准
                      cv=10,
                      n_jobs=-1)
    scores = cross_val_score(estimator=gs,X=X,y=y,scoring='accuracy',cv=5)
    print('cv accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))

    # 决策树模型使用嵌套交叉验证
    gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                      param_grid=[{'max_depth':[1,2,3,4,5,6,7,None]}],
                      scoring='accuracy',cv=5)

    scores = cross_val_score(gs,X=X,y=y,scoring='accuracy',cv=5)
    print('cv accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

    # 通过比较svm和决策树的输出结果可以得出结论，在此数据集上svm的性能优于决策树


############################了解不同的性能评价指标####################################################
# 混淆矩阵[[真正，假负],[假正，真负]]
def function8():
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC

    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
                     header=None)

    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(le.classes_)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))

    pipe_svc.fit(X_train,y_train)
    y_pred = pipe_svc.predict(X_test)
    confmat = confusion_matrix(y_true=y_test,y_pred=y_pred)
    print(confmat) # 打印混淆矩阵

    # 绘制混淆矩阵
    fig,ax = plt.subplots(figsize=(2.5,2.5))
    ax.matshow(confmat,cmap=plt.cm.Blues,alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=i,# x坐标
                    y=j,# y坐标
                    s=confmat[i,j], # 值
                    va='center',ha='center') # 显示位置
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.show()

# 准确率=真正/(真正+假正)  召回率=真正/(假负+真正)
# f1-分数 = 2*（真确率*召回率）/（准确率+召回率）
def function9():
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score,f1_score
    from sklearn.svm import SVC

    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
                     header=None)

    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(le.classes_)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))

    pipe_svc.fit(X_train, y_train)
    y_pred = pipe_svc.predict(X_test)

    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
    print('F1: %.3f' % f1_score( y_true = y_test, y_pred = y_pred))

    # 评分标准可以自己构造(模型默认以1为正类标)
    from sklearn.metrics import make_scorer
    scorer = make_scorer(f1_score,pos_label=0) # 构造以0为正类标，f1为分数参考
    # gs = GridSearchCV(estimator=pipe_svc,
    #                     param_grid=param_grid,
    #                     scoring=scorer,# 此处使用
    #                     cv=10)

# 绘制ROC曲线（受试者工作特征曲线）基于假正率和真正率等性能指标绘制
def function10():
    from sklearn.metrics import roc_curve,auc # auc为roc线下区域
    from scipy import interp # 一维线性插值函数
    from sklearn.model_selection import StratifiedKFold

    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
                     header=None)

    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(le.classes_)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    pipe_lr = make_pipeline(StandardScaler(),
                            PCA(n_components=2),
                            LogisticRegression(penalty='l2',random_state=1,C=100.0))

    X_train2 = X_train[:,[4,14]]
    cv = list(StratifiedKFold(n_splits=3,random_state=1).split(X_train,y_train))

    fig = plt.figure(figsize=(7,5))

    mean_tpr = 0.0
    mean_fpr = np.linspace(0,1,100)
    all_tpr = []

    for i,(train,test) in enumerate(cv):# 绘制三个分块的roc曲线
        probas = pipe_lr.fit(X_train2[train],y_train[train]).predict_proba(X_train2[test])
        fpr,tpr,thresholds = roc_curve(y_train[test],probas[:,1],pos_label=1)
        mean_tpr += interp(mean_fpr,fpr,tpr)
        mean_tpr[0] = 0.0 # 设置（0,0）坐标值为恒=0
        roc_auc = auc(fpr,tpr) # 根据假正率和真正率计算线下区域
        plt.plot(fpr,tpr,label='roc fold %d (area=%0.2f)' % (i+1,roc_auc))

    plt.plot([0,1],[0,1],linestyle='--', # 绘制随机猜测对应的roc曲线，线下区域为0.5
             color=(0.6,0.6,0.6),
             label='random guessing')

    mean_tpr /= len(cv) # 求真正率均值
    mean_tpr[-1] = 1.0 #设置（1,1）坐标值恒=1
    mean_auc = auc(mean_fpr,mean_tpr) # 计算auc均值

    plt.plot(mean_fpr,mean_tpr,'k--',
             label='mean roc (area=%0.2d)' % mean_auc,
             lw=2) # 线宽
    plt.plot([0,0,1],[0,1,1],linestyle=':',color='black',label='perfect performance')

    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel("true positive rate")
    plt.ylabel("false positive rate")
    plt.legend(loc='lower right')
    plt.show()

    # 可以直接通过roc_auc_score函数计算roc auc 得分
    from sklearn.metrics import roc_auc_score
    y_pred = pipe_lr.predict(X_test[:,[4,14]])
    print("roc auc: %.3f" %roc_auc_score(y_true=y_test,y_score=y_pred))

    # 最后，多类别分类的评价标准：宏均值和微均值，常用的有加权宏均值标准

# 数据集类别不均衡时的处理
def function11():
    from sklearn.utils import resample

    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
                     header=None)

    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values

    lbe = LabelEncoder()
    y = lbe.fit_transform(y)

    # 创建一个不均衡的数据集
    X_imb = np.vstack((X[y==0],X[y==1][:40]))
    y_imb = np.hstack((y[y==0],y[y==1][:40]))

    # 通过resample函数把少的类型样本的数量增加到和另一种一样多
    X_upsampled,y_upsampled = resample(X_imb[y_imb==1],
                                       y_imb[y_imb==1],
                                       replace=True,
                                       n_samples=X_imb[y_imb==0].shape[0])

    print(X_upsampled[y_upsampled==1].shape)
    print(X_imb[y_imb == 0].shape)





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
    # function5()
    # function6()
    # function7()
    # function8()
    # function9()
    # function10()
    function11()