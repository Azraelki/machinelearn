import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
'''
book-name:      machine learn 
book-location:  chapter-2
content:       自适应线性神经元实现--批梯度下降
'''
class AdalineGD:
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
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:])  + self.w_[0]

    # 激活函数，在这里仅做占位，并没有起作用
    def activation(self,X):
        return X

    # 预测函数
    def predict(self,X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


def show_cost(X,y):
    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,4))
    ada1 = AdalineGD(eta=0.01,n_iter=10).fit(X,y)
    ax[0].plot(range(1,len(ada1.cost_)+1),
               np.log10(ada1.cost_),marker="o")
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('adaline - learining rate 0.01')

    ada2 = AdalineGD(eta=0.0001,n_iter=10).fit(X,y)
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
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                     header=None)

    y = df.iloc[0:100, 4].values
    y = np.where(y == "Iris-setosa", -1, 1)

    X = df.iloc[0:100, [0, 2]].values

    ada = AdalineGD(n_iter=15,eta=0.01)
    ada.fit(standard_X(X,y),y)
    # show_cost(X,y)
    plot_decision_regions(standard_X(X,y),y,classifier=ada)
    plt.title('Adaline - gradient descent')
    plt.xlabel('sepal length [standartized]')
    plt.ylabel('petal length [standartized]')
    plt.legend(loc='upper left')
    plt.show()

    plt.plot(range(1, len(ada.cost_)+1),ada.cost_,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('sum - suquared - error')
    plt.show()
