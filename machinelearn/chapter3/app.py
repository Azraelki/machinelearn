from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
'''
sklearn学习
'''
def function1():
    # 获取数据集
    iris = datasets.load_iris()
    # 抽取数据集指定特征
    X = iris.data[:, [2, 3]]
    # 类标
    y = iris.target
    # 划分测试数据集合训练数据集
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)
    # print("labels counts in y:",np.bincount(y))
    # print("labels counts in y_train:",np.bincount(y_train))
    # print("labels counts in y_test:",np.bincount(y_test))

    # 标准化特征变量
    sc = StandardScaler()
    sc.fit(X,y)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # 使用sklearn内置的感知器模型训练数据
    ppn = Perceptron(n_iter=40,eta0=0.1, random_state=1)
    ppn.fit(X_train_std,y_train)
    # 使用测试数据集查看模型性能
    y_pred = ppn.predict(X_test_std)
    print("misclassfied sample: %d" % (y_test != y_pred).sum())
    print("Accuracy: %.2f" % accuracy_score(y_test,y_pred))

    X_combined_std = np.vstack((X_train_std,X_test_std))
    y_combined = np.hstack((y_train,y_test))
    plot_decision_regions(X=X_combined_std,y=y_combined,classifier=ppn,test_idx=range(105,150))
    plt.xlabel("petal length [standardized]")
    plt.ylabel("petal width [standardized]")
    plt.legend(loc='upper left')
    plt.show()

# 绘制决策区域
def plot_decision_regions(X,y,classifier,test_idx=None,resolution=0.02):
    # 初始化 标记符号和颜色集
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 绘制决策面
    x1_min,x1_max = X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max = X[:,1].min()-1,X[:,1].max()+1
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))

    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    # 绘制等高线图
    plt.contourf(xx1,xx2,Z,alpha=0.3,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    # 绘制样本的散点图
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

     # 高亮标识测试数据集的样本
    if test_idx:
        X_test,y_test = X[test_idx,:],y[test_idx]
        plt.scatter(X_test[:,0],X_test[:,1],
                    c='',edgecolors='black',alpha=1.0,
                    linewidth=1,marker='o',
                    s=100,label='test set')

def sigmod(z):
    return 1.0/(1.0+np.exp(-z))

# sigmod函数示例
def function2():
    z = np.arange(-7,7,0.1)
    phi_z = sigmod(z)
    plt.plot(z,phi_z)
    plt.axvline(0.0,color='red')
    plt.ylim(-0.1,1.1)
    plt.xlabel('z')
    plt.ylabel('$\phi (z)$')
    plt.yticks([0.0,0.5,1.0])
    ax = plt.gca()
    ax.yaxis.grid(True)
    plt.show()


# 使用sklearn实现逻辑斯蒂回归
def function3():
    # 获取数据集
    iris = datasets.load_iris()
    # 抽取数据集指定特征
    X = iris.data[:, [2, 3]]
    # 类标
    y = iris.target
    # 划分测试数据集合训练数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    # print("labels counts in y:",np.bincount(y))
    # print("labels counts in y_train:",np.bincount(y_train))
    # print("labels counts in y_test:",np.bincount(y_test))

    # 标准化特征变量
    sc = StandardScaler()
    sc.fit(X, y)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    lr = LogisticRegression(C=100.0,random_state=1)
    lr.fit(X_combined_std,y_combined)
    plot_decision_regions(X_combined_std,y_combined,classifier=lr,test_idx=range(105,150))

    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()
    print(lr.predict_proba(X_test_std[:3,:]))
    print(lr.predict_proba(X_test_std[:3, :]).sum(axis=1))
    print(lr.predict_proba(X_test_std[:3, :]).argmax(axis=1))
    print(lr.predict(X_test_std[:3, :]))
    print(X_test_std[0, :].reshape(1, -1))
    print(lr.predict(X_test_std[0, :].reshape(1,-1)))

# 正则化示例（解决欠拟合和过拟合）
def function4():
    # 获取数据集
    iris = datasets.load_iris()
    # 抽取数据集指定特征
    X = iris.data[:, [2, 3]]
    # 类标
    y = iris.target
    # 划分测试数据集合训练数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    # 标准化特征变量
    sc = StandardScaler()
    sc.fit(X, y)
    X_train_std = sc.transform(X_train)

    weights, params = [],[]
    for c in np.arange(-10,10):
        lr = LogisticRegression(C=100.**c, random_state=1)
        lr.fit(X_train_std,y_train)
        weights.append(lr.coef_[1])
        print(lr.coef_)
        params.append(10.**c)
    weights = np.array(weights)
    plt.plot(params, weights[:,0],label='petal length')
    plt.plot(params, weights[:,1],label='petal width',linestyle='--')
    plt.ylabel('wieght coefficent')
    plt.xlabel('C')
    plt.legend(loc='upper left')
    plt.xscale('log')
    plt.show()

# 使用线性SVM实现分类器
def function5():
    # 获取数据集
    iris = datasets.load_iris()
    # 抽取数据集指定特征
    X = iris.data[:, [2, 3]]
    # 类标
    y = iris.target
    # 划分测试数据集合训练数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    # 标准化特征变量
    sc = StandardScaler()
    sc.fit(X, y)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    svm = SVC(kernel='linear',C=1.0,random_state=1)
    svm.fit(X_train_std,y_train)
    plot_decision_regions(X_combined_std,y_combined,classifier=svm,test_idx=range(105,150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()

# 基于SGDClassifier构建模型（类似于随机梯度下降,当数据量过大时使用）
def function6():
    # 感知器模型
    ppn = SGDClassifier(loss='perceptron')
    # 逻辑斯谛回归模型
    lr = SGDClassifier(loss='log')
    # 支持向量机模型
    svm = SGDClassifier(loss='hinge')

# 非线性分类问题样例
def function7():
    # 随机生成一个非线性可分样本散点图
    np.random.seed(1)
    X_xor = np.random.randn(200,2)
    y_xor = np.logical_xor(X_xor[:,0] > 0,X_xor[:,1]>0)
    y_xor = np.where(y_xor,1,-1)
    plt.scatter(X_xor[y_xor==1,0],
                X_xor[y_xor==1,1],
                c='b',marker='x',
                label='1')
    plt.scatter(X_xor[y_xor == -1, 0],
                X_xor[y_xor == -1, 1],
                c='r', marker='o',
                label='-1')

    plt.xlim([-3,3])
    plt.ylim([-3,3])
    plt.legend(loc='best')
    plt.show()

# 非线性分类问题样例-svm
def function8():
    # 随机生成一个非线性可分样本散点图
    np.random.seed(1)
    X_xor = np.random.randn(200,2)
    y_xor = np.logical_xor(X_xor[:,0] > 0,X_xor[:,1]>0)
    y_xor = np.where(y_xor,1,-1)
    # 使用rbf(又称高斯核)核函数初始化svm,
    # 其中gamma是待优化自由参数,gamma越小，受影响的训练样本范围越大，决策边界越宽松
    svm = SVC(kernel='rbf',random_state=1,gamma=0.10,C=10.0)
    svm.fit(X_xor,y_xor)
    plot_decision_regions(X_xor,y_xor,classifier=svm)
    plt.legend(loc='best')
    plt.show()

# 使用基于rbf核函数实现SVM分类器
def function9():
    # 获取数据集
    iris = datasets.load_iris()
    # 抽取数据集指定特征
    X = iris.data[:, [2, 3]]
    # 类标
    y = iris.target
    # 划分测试数据集合训练数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    # 标准化特征变量
    sc = StandardScaler()
    sc.fit(X, y)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    svm = SVC(kernel='rbf',C=1.0,gamma=0.2,random_state=0)
    svm.fit(X_train_std,y_train)
    plot_decision_regions(X_combined_std,y_combined,classifier=svm,test_idx=range(105,150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()

# 决策树中不同信息增益在概率p下的的影响
# 基尼系数
def gini(p):
    return (p)*(1-(p)) + (1-p)*(1-(1-p))
# 熵
def entropy(p):
    return -p*np.log2(p) - (1-p)*np.log2((1-p))
# 误分类率
def error(p):
    return  1 - np.max([p,1-p])
def function10():
    x = np.arange(0.0,1.0,0.01)
    ent = [entropy(p) if p != 0 else None for p in x]
    sc_ent = [e*0.5 if e else None for e in ent]
    err = [error(i) for i in x]
    fig = plt.figure()
    ax = plt.subplot(111)
    for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],
                              ['Entropy', 'Entropy (scaled)',
                                'Gini Impurity',
                                'Misclassification Error'],
                                ['-', '-', '--', '-.'],
                              ['black', 'lightgray',
                                'red', 'green', 'cyan']):
        line = ax.plot(x, i, label=lab,linestyle = ls, lw = 2, color = c)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5,1.15),ncol=5,fancybox=True,shadow=False)
    ax.axhline(y=0.5,linewidth=1,color='k',linestyle='--')
    ax.axhline(y=1.0,linewidth=1,color='k',linestyle='--')
    plt.ylim([0,1.1])
    plt.xlabel("p(i=1)")
    plt.ylabel("impurity index")
    plt.show()

# sklearn构建决策树
def function11():
    # 获取数据集
    iris = datasets.load_iris()
    # 抽取数据集指定特征
    X = iris.data[:, [2, 3]]
    # 类标
    y = iris.target
    # 划分测试数据集合训练数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    # 标准化特征变量
    sc = StandardScaler()
    sc.fit(X, y)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    # 信息增益使用基尼系数标准
    tree = DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=1)

    tree.fit(X_train_std,y_train)
    plot_decision_regions(X_combined_std,y_combined,classifier=tree,test_idx=range(105,150))
    plt.xlabel("petal length [cm]")
    plt.ylabel("petal width [cm]")
    plt.legend(loc="upper left")
    plt.show()

    # 导出为.dot文件，可视化决策树,需要安装报错的程序
    from pydotplus import graph_from_dot_data
    from sklearn.tree import export_graphviz
    dot_data = export_graphviz(tree,filled=True,
                               class_names=['Setosa','Versicolor','Virginica'],
                               feature_names=['petal length','petal width'],
                               out_file=None)
    graph = graph_from_dot_data(dot_data)
    graph.write_png('tree.png')

# 使用sklearn构造随机森林模型
def function12():
    from sklearn.ensemble import RandomForestClassifier

    # 获取数据集
    iris = datasets.load_iris()
    # 抽取数据集指定特征
    X = iris.data[:, [2, 3]]
    # 类标
    y = iris.target
    # 划分测试数据集合训练数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    # 标准化特征变量
    sc = StandardScaler()
    sc.fit(X, y)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))


    forest = RandomForestClassifier(criterion='gini',
                                    n_estimators=25,# 创建的决策树数量
                                    random_state=1,
                                    n_jobs=2) # 使用的处理器内核数量
    forest.fit(X_train_std,y_train)
    plot_decision_regions(X_combined_std,y_combined,classifier=forest,test_idx=range(105,150))

    plt.xlabel("petal length [cm]")
    plt.ylabel("petal width [cm]")
    plt.legend(loc="upper left")
    plt.show()

# 使用sklearn实现一个KNN模型
def function13():
    from sklearn.neighbors import KNeighborsClassifier

    # 获取数据集
    iris = datasets.load_iris()
    # 抽取数据集指定特征
    X = iris.data[:, [2, 3]]
    # 类标
    y = iris.target
    # 划分测试数据集合训练数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    # 标准化特征变量
    sc = StandardScaler()
    sc.fit(X, y)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    knn = KNeighborsClassifier(n_neighbors=5, # 选择的邻居数量
                               p=1, # 与metric参数相呼应，当p=1时为曼哈顿距离，当p=2为欧几里得距离
                               metric='minkowski')# 闵可夫斯基距离，是曼哈顿距离和欧几里得距离的泛华
    knn.fit(X_train_std,y_train)
    plot_decision_regions(X_combined_std,y_combined,
                          classifier=knn,test_idx=range(105,150))

    plt.xlabel("petal length [cm]")
    plt.ylabel("petal width [cm]")
    plt.legend(loc="upper left")
    plt.show()





if __name__ == '__main__':
    # 获取数据集
    iris = datasets.load_iris()
    # 抽取数据集指定特征
    X = iris.data[:, [2, 3]]
    # 类标
    y = iris.target
    # 划分测试数据集合训练数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    # 标准化特征变量
    sc = StandardScaler()
    sc.fit(X, y)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))


    # function1()
    # function2()
    # function3()
    # function4()
    # function5()
    # function7()
    # function8()
    # function9()
    # function10()
    # function11()
    # function12()
    function13()