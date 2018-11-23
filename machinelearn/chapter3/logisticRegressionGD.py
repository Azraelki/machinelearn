import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
'''
book-name:      machine learn 
book-location:  chapter-2
content:       逻辑斯谛回归模型实现--批梯度下降
'''
class LogisticRegressionGD:
    def __init__(self, eta=0.01,n_iter=50,random_state=1):
        '''
        :param eta: 学习速率，介于0-1之间
        :param n_iter: 迭代次数
        :param random_state: 用于初始化权重参数的随机种子，用于复现测试结果
        '''
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self,X,y):
        '''
        :param X: 变量，np.array类型
        :param y: 因变量，np.array类型
        :return:
        '''
        # 初始化权重参数
        rgen = np.random.RandomState(self.random_state)
        self.w_  =  rgen.normal(loc=0.0, scale=0.01,size = 1+X.shape[1])

        # 成本函数返回值列表初始化
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y-output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            # 修改成本函数为逻辑斯谛模型的成本函数
            # 原有的成本函数：cost = (errors**2).sum()/2.0
            cost = (-y.dot(np.log(output))-((1-y).dot(np.log(1-output))))
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:])  + self.w_[0]

    # 激活函数，把净输出投影到sigmod函数上
    def activation(self,X):
        return 1.0 / (1.0 + np.exp(-np.clip(X,-250,250)))

    # 预测函数
    def predict(self,X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


def show_cost(X,y):
    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,4))
    ada1 = LogisticRegressionGD(eta=0.01,n_iter=10).fit(X,y)
    ax[0].plot(range(1,len(ada1.cost_)+1),
               np.log10(ada1.cost_),marker="o")
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('adaline - learining rate 0.01')

    ada2 = LogisticRegressionGD(eta=0.0001,n_iter=10).fit(X,y)
    ax[1].plot(range(1, len(ada2.cost_) + 1),
               np.log10(ada2.cost_), marker="o")
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('log(Sum-squared-error)')
    ax[1].set_title('adaline - learining rate 0.0001')

    plt.show()

def standard_X(X,y):
    X_std = np.copy(X)
    X_std[:,0] = (X[:,0]-X[:,0].mean())/X[:,0].std()
    X_std[:,1] = (X[:,1]-X[:,1].mean())/X[:,1].std()

    return X_std

def plot_decision_regions(X,y,classifier,resolution=0.02):

    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max,resolution),
                           np.arange(x2_min,x2_max,resolution))
    z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,z,alpha=0.3,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],
                    y=X[y==cl,1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')








if __name__ == "__main__":
    # 获取数据集
    iris = datasets.load_iris()
    # 抽取数据集指定特征
    X = iris.data[:, [2, 3]]
    # 类标
    y = iris.target

    # 提取两个特征的样本
    X = X[(y==0) | (y==1)]
    y = y[(y==0) | (y==1)]

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

    ada = LogisticRegressionGD(n_iter=1000, eta=0.05,random_state=1)
    ada.fit(X_combined_std, y_combined)
    # show_cost(X,y)
    plot_decision_regions(X_combined_std, y_combined, classifier=ada)
    plt.title('Adaline - gradient descent')
    plt.xlabel('sepal length [standartized]')
    plt.ylabel('petal length [standartized]')
    plt.legend(loc='upper left')
    plt.show()

    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('sum - suquared - error')
    plt.show()