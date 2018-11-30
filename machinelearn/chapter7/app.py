import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.misc import comb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from MajorityVowteClassifier import MajoritVoteClassifier

# 获取数据并且将y值量化
iris = datasets.load_iris()
X,y = iris.data[50:,[1,2]],iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)
# 划分数据集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5,random_state=1,stratify=y)

def ensemble_error(n_classifier,error):
    '''
    集成分类器误差率计算
    :param n_classifier: 分类器个数
    :param error:  单个分类器的误差率
    :return: 集成分类器误差率
    '''
    k_start = int(math.ceil(n_classifier/2.))
    probs = [comb(n_classifier,k)*(error**k)*(1-error)**(n_classifier-k)
             for k in range(k_start,n_classifier+1)]
    return sum(probs)

# 绘制集成分类器和单个分类器随着单个分类器的误差率变化而变化的曲线
def function1():
    error_range = np.arange(0.0,1.01,0.1)
    ens_errors = [ensemble_error(n_classifier=11,error=error) for error in error_range]

    plt.plot(error_range,error_range,linestyle='--',label='base error',lw='2')
    plt.plot(error_range,ens_errors,label='ensemble error',lw='2')
    plt.xlabel("base error")
    plt.ylabel("base/ensemble error")
    plt.legend(loc='upper left')
    plt.grid(alpha=0.5)
    plt.show()

# 使用k-折交叉验证评估各评估器的性能
def function2():
    clf1 = LogisticRegression(penalty='l2',C=0.001,random_state=1)
    clf2 = DecisionTreeClassifier(max_depth=1,criterion='entropy',random_state=0)
    clf3 = KNeighborsClassifier(n_neighbors=1,p=2,metric='minkowski')
    pipe1 = Pipeline([['sc',StandardScaler()],['clf',clf1]])
    pipe3 = Pipeline([['sc',StandardScaler()],['clf',clf3]])

    mv_clf = MajoritVoteClassifier(classifiers=[pipe1,clf2,pipe3])

    clf_labels = ['logistic regression','decision tree','KNN','majority voting']
    all_clf = [pipe1,clf2,pipe3,mv_clf]
    print("10-fold cross validation:\n")
    for clf,label in zip(all_clf,clf_labels):
        scores = cross_val_score(estimator=clf,X=X_train,y=y_train,
                                 cv=10,scoring='roc_auc')
        print("roc auc: %0.2f (+/- %0.2f) [%s]"%(scores.mean(),scores.std(),label))

    # 由打印的结果可以看出，集成分类器的性能相对于单个的分类器有质的提升

    # 绘制评估器的roc曲线，评估模型性能
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc

    colors = ['black','orange','blue','green']
    linestyle = [':','--','-.','-']
    for clf, label,clr,ls in zip(all_clf,clf_labels,colors,linestyle):
        y_pred = clf.fit(X_train,y_train).predict_proba(X_test)[:,1] # 假定正分类为1
        fpr,tpr,thresholds = roc_curve(y_true=y_test,y_score=y_pred)
        roc_auc = auc(x=fpr,y=tpr)
        plt.plot(fpr,tpr,color=clr,linestyle=ls,label='%s (auc=%.2f)' % (label,roc_auc))

    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],linestyle='--',color='gray',lw=2) # 随机猜测roc曲线
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.grid(alpha=0.5)
    plt.xlabel('false positive rate (fpr)')
    plt.ylabel('true positive rate (tpr)')
    plt.show()

    # 绘制各个评估器的决策边界
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    from itertools import product
    x1_min = X_train_std[:,0].min()-1
    x1_max = X_train_std[:,0].max()+1
    x2_min = X_train_std[:,1].min()-1
    x2_max = X_train_std[:,1].max()+1
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,0.1),
                          np.arange(x2_min,x2_max,0.1))
    f,axarr = plt.subplots(nrows=2,ncols=2,
                           sharex='col',sharey='row',
                           figsize=(7,5))
    for idx,clf,tt in zip(product([0,1],[0,1]),all_clf,clf_labels):
        clf.fit(X_train_std,y_train)
        Z = clf.predict(np.c_[xx1.ravel(),xx2.ravel()])
        Z = Z.reshape(xx1.shape)
        axarr[idx[0],idx[1]].contourf(xx1,xx2,Z,alpha=0.3)
        axarr[idx[0],idx[1]].scatter(X_train_std[y_train==0,0],
                                     X_train_std[y_train==0,1],
                                     c='blue',marker='^',s=50)
        axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 1, 0],
                                      X_train_std[y_train == 1, 1],
                                      c='green', marker='o', s=50)
        axarr[idx[0],idx[1]].set_title(tt)

    plt.text(-3.5,-4.5,s='sepal width [standardized]',ha='center',va='center',fontsize=12)
    plt.text(-12.5,4.5,s='sepal length [standardized]',ha='center',va='center',fontsize=12,rotation=90)
    plt.show()

# 集成分类的成员分类器调优
def function3():
    clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=1)
    clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
    clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
    pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
    pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])

    mv_clf = MajoritVoteClassifier(classifiers=[pipe1, clf2, pipe3])

    # 打印出参数构成
    print(mv_clf.get_params())

    # 使用网格搜索对决策树深度，逻辑斯谛回归的正则参数进行调优
    from sklearn.model_selection import GridSearchCV
    params = {
                'decisiontreeclassifier__max_depth':[1,2],
                'pipeline-1__clf__C':[0.001,0.1,100.0]
              }
    grid = GridSearchCV(estimator=mv_clf,param_grid=params,cv=10,scoring='roc_auc')
    grid.fit(X_train,y_train)

    print("best parameters:%s" % grid.best_params_)
    print("accuracy : %.2f" % grid.best_score_)










if __name__ == "__main__":
    # function1()
    # function2()
    function3()